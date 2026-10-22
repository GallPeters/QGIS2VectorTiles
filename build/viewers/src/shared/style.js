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

import { PMTiles } from 'pmtiles';

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

function archiveAt(url) {
    let archive = archives.get(url);
    if (!archive) {
        archive = new PMTiles(url);
        archives.set(url, archive);
    }
    return archive;
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
