"""
pmtiles_writer.py

mbtiles_to_pmtiles() — copies the tiles of an MBTiles file, as they are (no
re-encoding), into one PMTiles v3 archive: a single file any web server or
static host (e.g. GitHub Pages) serves with HTTP range requests, which the
viewers read directly.

Layout written: header (127 bytes), root directory, JSON metadata, leaf
directories, tile data. Tiles are stored in Hilbert tile-id order
(clustered); identical tiles are stored once, and runs of them with
consecutive ids share one directory entry. Directories and metadata are
gzip-compressed, as the specification recommends.

Specification: https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md

Depends on: nothing beyond the standard library
"""

import gzip
import hashlib
import json
import os
import sqlite3
import struct
from typing import Dict, List, Tuple

_HEADER_SIZE = 127
# The header and root directory must fit in the first 16 KiB, which readers
# fetch in one request.
_ROOT_MAX = 16384 - _HEADER_SIZE

# Header codes.
_COMPRESSION_NONE, _COMPRESSION_GZIP = 1, 2
_TILE_TYPE_MVT = 1

# An entry: (tile id, offset in the tile data or leaf section, length,
# run length; 0 = a leaf directory).
_Entry = Tuple[int, int, int, int]


def zxy_to_tile_id(z: int, x: int, y: int) -> int:
    """The PMTiles tile id: tiles of lower zooms first, then the position
    along the zoom level's Hilbert curve."""
    acc = ((1 << z) * (1 << z) - 1) // 3
    n = z - 1
    tx, ty = x, y
    a = 1 << n if z > 0 else 0
    while a > 0:
        rx = tx & a
        ry = ty & a
        acc += ((3 * rx) ^ ry) * (1 << n)
        if ry == 0:
            if rx != 0:
                tx, ty = a - 1 - tx, a - 1 - ty
            tx, ty = ty, tx
        a >>= 1
        n -= 1
    return acc


def _varint(value: int, out: bytearray) -> None:
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _serialize_directory(entries: List[_Entry]) -> bytes:
    """A directory, gzip-compressed: the entry count, then the ids (as
    deltas), run lengths, lengths and offsets (0 when an entry follows the
    previous one directly, else offset + 1)."""
    out = bytearray()
    _varint(len(entries), out)
    last_id = 0
    for tile_id, _, _, _ in entries:
        _varint(tile_id - last_id, out)
        last_id = tile_id
    for _, _, _, run_length in entries:
        _varint(run_length, out)
    for _, _, length, _ in entries:
        _varint(length, out)
    for i, (_, offset, _, _) in enumerate(entries):
        previous = entries[i - 1] if i > 0 else None
        if previous is not None and offset == previous[1] + previous[2]:
            _varint(0, out)
        else:
            _varint(offset + 1, out)
    return gzip.compress(bytes(out), mtime=0)


def _build_directories(entries: List[_Entry]) -> Tuple[bytes, bytes]:
    """(root directory, leaf directories): all entries in the root if they
    fit in the first 16 KiB, else in leaves the root points to, as few as
    keep the root small enough."""
    root = _serialize_directory(entries)
    if len(root) <= _ROOT_MAX:
        return root, b""
    leaf_size = 4096
    while True:
        root_entries: List[_Entry] = []
        leaves = bytearray()
        for start in range(0, len(entries), leaf_size):
            chunk = entries[start:start + leaf_size]
            leaf = _serialize_directory(chunk)
            root_entries.append((chunk[0][0], len(leaves), len(leaf), 0))
            leaves += leaf
        root = _serialize_directory(root_entries)
        if len(root) <= _ROOT_MAX:
            return root, bytes(leaves)
        leaf_size = int(leaf_size * 1.2)


def _metadata(meta: Dict[str, str]) -> dict:
    """The archive's JSON metadata from the MBTiles metadata table: its
    name, attribution and so on, and the vector layers."""
    result = {
        key: value for key, value in meta.items()
        if key not in ("json", "bounds", "center", "minzoom", "maxzoom", "format")
    }
    try:
        result.update(json.loads(meta.get("json", "{}")))
    except ValueError:
        pass
    return result


def _e7(value: float) -> int:
    return int(round(value * 10_000_000))


def mbtiles_to_pmtiles(mbtiles_path: str, pmtiles_path: str) -> int:
    """Write the tiles of ``mbtiles_path`` to a PMTiles archive at
    ``pmtiles_path``; returns the number of tiles."""
    conn = sqlite3.connect(f"file:{mbtiles_path}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT name, value FROM metadata"))
        # MBTiles rows count from the south (TMS); PMTiles from the north.
        # Tiles are then read by rowid, which needs no index.
        rows = sorted(
            (zxy_to_tile_id(z, x, (1 << z) - 1 - y), z, rowid)
            for rowid, z, x, y in conn.execute(
                "SELECT rowid, zoom_level, tile_column, tile_row FROM tiles"
            )
        )

        temp_path = f"{pmtiles_path}.tmp"
        tile_data_path = f"{pmtiles_path}.tiles.tmp"
        entries: List[_Entry] = []
        offsets_by_hash: Dict[bytes, Tuple[int, int]] = {}
        tile_data_length = 0
        tile_compression = _COMPRESSION_NONE
        min_zoom, max_zoom = 255, 0
        try:
            with open(tile_data_path, "wb") as tile_data:
                for i, (tile_id, z, rowid) in enumerate(rows):
                    (data,) = conn.execute(
                        "SELECT tile_data FROM tiles WHERE rowid=?", (rowid,)
                    ).fetchone()
                    data = bytes(data)
                    if i == 0 and data[:2] == b"\x1f\x8b":
                        tile_compression = _COMPRESSION_GZIP
                    min_zoom, max_zoom = min(min_zoom, z), max(max_zoom, z)
                    digest = hashlib.sha256(data).digest()
                    known = offsets_by_hash.get(digest)
                    if known is None:
                        known = (tile_data_length, len(data))
                        offsets_by_hash[digest] = known
                        tile_data.write(data)
                        tile_data_length += len(data)
                    last = entries[-1] if entries else None
                    if (last is not None and last[1:3] == known
                            and last[0] + last[3] == tile_id):
                        entries[-1] = (last[0], last[1], last[2], last[3] + 1)
                    else:
                        entries.append((tile_id, known[0], known[1], 1))

            root, leaves = _build_directories(entries)
            metadata = gzip.compress(
                json.dumps(_metadata(meta), separators=(",", ":")).encode("utf-8"), mtime=0
            )
            root_offset = _HEADER_SIZE
            metadata_offset = root_offset + len(root)
            leaves_offset = metadata_offset + len(metadata)
            tile_data_offset = leaves_offset + len(leaves)

            try:
                west, south, east, north = (float(v) for v in meta["bounds"].split(","))
            except (KeyError, ValueError):
                west, south, east, north = -180.0, -85.0511287, 180.0, 85.0511287
            try:
                center_lon, center_lat, center_zoom = (float(v) for v in meta["center"].split(","))
            except (KeyError, ValueError):
                center_lon, center_lat = (west + east) / 2, (south + north) / 2
                center_zoom = min_zoom
            if not rows:
                min_zoom = max_zoom = 0

            header = b"PMTiles" + struct.pack(
                "<BQQQQQQQQQQQBBBBBBiiiiBii",
                3,
                root_offset, len(root),
                metadata_offset, len(metadata),
                leaves_offset, len(leaves),
                tile_data_offset, tile_data_length,
                len(rows), len(entries), len(offsets_by_hash),
                1,                       # clustered
                _COMPRESSION_GZIP,       # internal compression
                tile_compression,
                _TILE_TYPE_MVT,
                min_zoom, max_zoom,
                _e7(west), _e7(south), _e7(east), _e7(north),
                int(round(center_zoom)), _e7(center_lon), _e7(center_lat),
            )
            assert len(header) == _HEADER_SIZE
            with open(temp_path, "wb") as out:
                out.write(header)
                out.write(root)
                out.write(metadata)
                out.write(leaves)
                with open(tile_data_path, "rb") as tile_data:
                    while True:
                        chunk = tile_data.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            os.replace(temp_path, pmtiles_path)
        finally:
            for path in (temp_path, tile_data_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return len(rows)
    finally:
        conn.close()


__all__ = ["mbtiles_to_pmtiles", "zxy_to_tile_id"]
