"""
base_layer_cache.py

BaseLayerCache — keeps prepared base layers (fixed, reprojected, exploded
and simplified copies of the project sources) between runs, so a source is
only read and prepared again when it, the export extent or the fields the
rules need have changed.

An entry's key covers everything its content depends on: provider, source
URI, extent, required fields, the size and modification time of every file
of the source, the preparation settings and CACHE_VERSION. Only sources read
from local files are cached. Entries are written under a temporary name and
renamed when complete, so an interrupted run never leaves a broken entry.

Depends on: config (constants only)
"""

import hashlib
import json
import os
import tempfile
import time
from glob import escape, glob
from os.path import basename, exists, isfile, join, splitext
from typing import List, Optional, Tuple
from uuid import uuid4

from ..utils.config import (
    _CACHE_DIR_NAME,
    _CACHE_MAX_AGE_DAYS,
    _CACHE_MAX_MB,
    _DATA_SIMPLIFICATION_TOLERANCE,
    _EPSG_CRS,
)

# Bump whenever the way base layers are built changes, to invalidate entries
# made by older code.
CACHE_VERSION = 2

# Providers whose URI starts with a local file path.
_FILE_PROVIDERS = frozenset({"ogr"})


class BaseLayerCache:
    """File cache of prepared base layers, shared by all runs on this machine."""

    def __init__(self, extension: str, directory: Optional[str] = None):
        self.extension = extension
        self.directory = directory or join(tempfile.gettempdir(), _CACHE_DIR_NAME)
        self.enabled = _CACHE_MAX_MB > 0
        if self.enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except OSError:
                self.enabled = False

    # --- keys ---

    def path_for(self, src) -> Optional[str]:
        """Cache path for a source snapshot, or None if it can't be cached."""
        if not self.enabled or src.provider not in _FILE_PROVIDERS:
            return None
        files = self._source_files(src.source_uri)
        if not files:
            return None
        try:
            signature = [
                (basename(f), os.stat(f).st_size, os.stat(f).st_mtime_ns) for f in files
            ]
        except OSError:
            return None
        key = json.dumps([
            CACHE_VERSION, src.provider, src.source_uri,
            None if src.extent is None else [repr(v) for v in src.extent],
            None if src.required_fields is None else sorted(src.required_fields),
            signature, _EPSG_CRS, _DATA_SIMPLIFICATION_TOLERANCE,
        ])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return join(self.directory, f"base_{digest}.{self.extension}")

    @staticmethod
    def _source_files(uri: str) -> List[str]:
        """Every local file the source reads, or [] if that can't be told."""
        path = uri.split("|")[0]
        if not isfile(path):
            return []
        stem = splitext(path)[0]
        # Shapefile-style sidecars (.dbf, .shx, .prj, .cpg, ...) and
        # SQLite/GeoPackage journals (-wal, -shm).
        siblings = glob(escape(stem) + ".*") + glob(escape(path) + "-*")
        return sorted(set([path] + [f for f in siblings if isfile(f)]))

    # --- entries ---

    def hit(self, path: str) -> bool:
        """True if a complete entry exists; marks it as recently used."""
        if not exists(path):
            return False
        try:
            os.utime(path)
        except OSError:
            pass
        return True

    @staticmethod
    def temp_path_for(path: str) -> str:
        """Where to build an entry before commit()."""
        stem, ext = splitext(path)
        return f"{stem}.tmp_{uuid4().hex}{ext}"

    @staticmethod
    def commit(temp_path: str, path: str) -> str:
        """Move a finished entry into place; returns the path to read."""
        try:
            os.replace(temp_path, path)
        except OSError:
            # Another run committed it first and has it open: use theirs.
            if exists(path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            else:
                return temp_path
        return path

    # --- housekeeping ---

    def prune(self) -> None:
        """Remove stale temporaries, entries unused for too long, and the
        least recently used entries beyond the size limit."""
        if not self.enabled:
            return
        now = time.time()
        entries: List[Tuple[float, int, str]] = []
        for path in glob(join(self.directory, "base_*")):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            is_temp = ".tmp_" in basename(path)
            too_old = now - stat.st_mtime > (1 if is_temp else _CACHE_MAX_AGE_DAYS) * 86400
            if too_old:
                self._remove(path)
            elif not is_temp:
                entries.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in entries)
        limit = _CACHE_MAX_MB * 1024 * 1024
        for _, size, path in sorted(entries):
            if total <= limit:
                break
            if self._remove(path):
                total -= size

    @staticmethod
    def _remove(path: str) -> bool:
        try:
            os.remove(path)
            return True
        except OSError:
            return False  # In use by another run.


__all__ = ["BaseLayerCache", "CACHE_VERSION"]
