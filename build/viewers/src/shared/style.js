// Loads the export's style.json for either map library. Its URLs are
// relative to style.json (so an export works from any web server or static
// host, e.g. GitHub Pages) and are made absolute here; the tiles are one
// PMTiles file, read with HTTP range requests.
//
// The PMTiles source becomes a plain vector source with the archive's zoom
// levels and bounds, whose tile URLs each viewer chooses (tileUrls):
// MapLibre reads pmtiles://<archive>/{z}/{x}/{y} with the pmtiles
// protocol, OpenLayers <archive>#pmtiles={z}/{x}/{y} with readPmtilesTile()
// (its style loader rewrites URLs with another scheme).

import { FetchSource, PMTiles } from 'pmtiles';

const PMTILES_PREFIX = 'pmtiles://';
const OL_TILE = /^(.+)#pmtiles=(\d+)\/(\d+)\/(\d+)$/;

/** Open archives by URL, shared by the style and the tile requests. */
export const archives = new Map();

// The style's popup descriptions: {source-layer: {q2vt_rule or "": text}}.
let descriptions = {};

/** A feature's properties with its rule's description (q2vt_description),
 * which the style keeps once per rule rather than the tiles per feature. */
export function withDescription(sourceLayer, properties) {
    const rules = descriptions[sourceLayer];
    if (!rules || properties.q2vt_description !== undefined) return properties;
    const text = rules[properties.q2vt_rule ?? ''] ?? rules[''];
    return text === undefined ? properties : { q2vt_description: text, ...properties };
}

// The archive's bytes read so far, kept in the browser between visits
// (Cache Storage), by the archive's URL: its version (?v=, unique per
// export) makes a new export a new archive, and its older versions' are
// removed. A returning visitor's map then asks the server for nothing it
// had, not even whether it changed.
const STORE_PREFIX = 'q2vt-tiles:';

async function openStore(url) {
    if (!/[?&]v=/.test(url) || typeof caches === 'undefined') return null;  // Unversioned, or not https/localhost.
    try {
        const name = STORE_PREFIX + url;
        const archive = (address) => { const u = new URL(address); return [u.origin + u.pathname, u.searchParams.get('v')]; };
        const [path, version] = archive(url);
        // The archive's other versions (earlier exports), removed meanwhile.
        caches.keys().then((names) => names.forEach((other) => {
            if (!other.startsWith(STORE_PREFIX)) return;
            const [otherPath, otherVersion] = archive(other.slice(STORE_PREFIX.length));
            if (otherPath === path && otherVersion !== version) caches.delete(other);
        })).catch(() => {});
        return await caches.open(name);
    } catch (error) {
        return null;  // Storage blocked (private mode, settings): read from the server.
    }
}

/** A PMTiles source reading byte ranges from the browser's store, else
 * from the server (and keeping them). */
class StoredSource {
    constructor(url) {
        this.url = url;
        this.server = new FetchSource(url);
        this.store = openStore(url);
    }

    getKey() {
        return this.url;
    }

    async getBytes(offset, length, signal, etag) {
        const store = await this.store;
        if (!store) return this.server.getBytes(offset, length, signal, etag);
        // Nothing kept yet (a first visit): no store look-ups to wait for.
        if (this.empty === undefined) this.empty = store.keys().then((keys) => keys.length === 0).catch(() => false);
        if (await this.empty) return this.keep(store, offset, length, signal, etag);
        const key = `${this.url}${this.url.includes('?') ? '&' : '?'}q2vt-bytes=${offset}-${length}`;
        const kept = await store.match(key).catch(() => undefined);
        if (kept) return { data: await kept.arrayBuffer(), etag: kept.headers.get('ETag') || undefined };
        return this.keep(store, offset, length, signal, etag);
    }

    /** The bytes from the server, kept in the store. */
    async keep(store, offset, length, signal, etag) {
        const response = await this.server.getBytes(offset, length, signal, etag);
        const key = `${this.url}${this.url.includes('?') ? '&' : '?'}q2vt-bytes=${offset}-${length}`;
        const headers = response.etag ? { ETag: response.etag } : {};
        store.put(key, new Response(response.data.slice(0), { headers })).catch(() => {});
        return response;
    }
}

function archiveAt(url) {
    let archive = archives.get(url);
    if (!archive) {
        archive = new PMTiles(new StoredSource(url));
        archives.set(url, archive);
        prefetchLeaves(archive);
    }
    return archive;
}

// Leaf directories read up front when they are at most this many and this
// large (the library caches 100 directories).
const PREFETCH_LEAVES = { count: 64, bytes: 1 << 20 };

/** Reads an archive's leaf directories at once, in parallel. A tile listed in
 * a leaf otherwise waits for its leaf first: two requests one after the other
 * for each new area, which on a slow connection is most of the wait. */
async function prefetchLeaves(archive) {
    try {
        const header = await archive.getHeader();
        if (!header.leafDirectoryLength || header.leafDirectoryLength > PREFETCH_LEAVES.bytes) return;
        const { cache, source } = archive;
        const root = await cache.getDirectory(source, header.rootDirectoryOffset, header.rootDirectoryLength, header);
        const leaves = root.filter((entry) => entry.runLength === 0);
        if (leaves.length > PREFETCH_LEAVES.count) return;
        await Promise.all(leaves.map((entry) => cache.getDirectory(
            source, header.leafDirectoryOffset + entry.offset, entry.length, header)));
    } catch (error) {
        // The tiles still load, each leaf when first needed.
    }
}

/** Tile URL templates of an archive, per viewer. */
export const tileUrls = {
    maplibre: (url) => `${PMTILES_PREFIX}${url}/{z}/{x}/{y}`,
    openlayers: (url) => `${url}#pmtiles={z}/{x}/{y}`,
};

/** ``url`` made absolute against ``base``; {placeholders} are kept as they
 * are (URL() would escape their braces). */
function resolveUrl(url, base) {
    const brace = url.indexOf('{');
    const head = brace < 0 ? url : url.slice(0, brace);
    const tail = brace < 0 ? '' : url.slice(brace);
    return new URL(head, base).href + tail;
}

/** The style at ``styleUrl`` (relative to the page), ready for the map;
 * ``tileUrl(archiveUrl)`` gives the PMTiles source's tile URL template. */
export async function loadStyle(styleUrl, tileUrl) {
    const base = new URL(styleUrl, window.location.href).href;
    const response = await fetch(base);
    if (!response.ok) throw new Error(`Cannot load the style (${response.status}): ${base}`);
    const style = await response.json();
    descriptions = style.metadata?.['q2vt:descriptions'] || {};
    if (typeof style.glyphs === 'string') style.glyphs = resolveUrl(style.glyphs, base);
    if (typeof style.sprite === 'string') style.sprite = resolveUrl(style.sprite, base);
    for (const [id, source] of Object.entries(style.sources || {})) {
        if (typeof source.url === 'string' && source.url.startsWith(PMTILES_PREFIX)) {
            const url = resolveUrl(source.url.slice(PMTILES_PREFIX.length), base);
            const header = await archiveAt(url).getHeader();
            const { url: _, ...rest } = source;
            style.sources[id] = {
                ...rest,
                tiles: [tileUrl(url)],
                minzoom: header.minZoom,
                maxzoom: header.maxZoom,
                bounds: [header.minLon, header.minLat, header.maxLon, header.maxLat],
            };
        } else if (Array.isArray(source.tiles)) {
            source.tiles = source.tiles.map((tile) => resolveUrl(tile, base));
        } else if (typeof source.url === 'string') {
            source.url = resolveUrl(source.url, base);
        }
    }
    return style;
}

/** The tile behind an OpenLayers PMTiles tile URL, as a Response (empty
 * where the archive has no tile); null for other URLs. */
export function readPmtilesTile(url) {
    const match = OL_TILE.exec(url);
    if (!match) return null;
    const [, archiveUrl, z, x, y] = match;
    return archiveAt(archiveUrl).getZxy(+z, +x, +y)
        .then((tile) => new Response(tile ? tile.data : new ArrayBuffer(0)));
}
