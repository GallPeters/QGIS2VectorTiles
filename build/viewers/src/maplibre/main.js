// MapLibre viewer: the map library adapter for the shared controls.
// maplibre-gl.mjs is not bundled; it sits next to bundle.js and loads its
// worker (maplibre-gl-worker.mjs) from the same folder.

import * as maplibregl from './maplibre-gl.mjs';
import { Protocol } from 'pmtiles';
import { startViewer } from '../shared/app.js';
import { drawBadge } from '../shared/badge.js';
import { CREDIT_HTML } from '../shared/credit.js';
import { localized, nameExpression, openFreeMapStyle, showsNames } from '../shared/openfreemap.js';
import { archives, LIVE_SOURCE, liveTilesUrl, loadStyle, refreshLive, tileUrls, withDescription } from '../shared/style.js';
import '../shared/ui.css';

const SOURCE = 'q2vt_tiles';
const BASEMAP = 'q2vt-basemap';
// OpenFreeMap's layers, sources, fonts and icons, by this prefix.
const OFM = 'ofm:';
const GLYPHS = 'q2vt-glyphs';
const HIGHLIGHT = 'q2vt-search-hit';
const config = window.Q2VT_CONFIG;

// Right-to-left labels (Hebrew, Arabic) need MapLibre's RTL text plugin to
// show in reading order and, for Arabic, with joined letters. It sits next
// to bundle.js and loads only when such a label first appears.
maplibregl.setRTLTextPlugin(new URL('mapbox-gl-rtl-text.js', import.meta.url).href, true)
    .catch((error) => console.warn('RTL text plugin not loaded:', error));

// The tiles are read from the export's PMTiles file; the protocol reuses
// the archives the style loader opened (and their cached directories).
const liveUrl = liveTilesUrl(config);
const style = await loadStyle(config.styleUrl, tileUrls.maplibre, liveUrl);
const protocol = new Protocol();
for (const archive of archives.values()) protocol.add(archive);
maplibregl.addProtocol('pmtiles', protocol.tile);

// A map has one glyphs address: the export's fonts, and OpenFreeMap's for
// its labels (their font names marked OFM, see addOpenFreeMap), each from
// its own place.
const projectGlyphs = style.glyphs;
let openFreeMapGlyphs = null;
style.glyphs = `${GLYPHS}://{fontstack}/{range}`;
maplibregl.addProtocol(GLYPHS, async (params, abortController) => {
    const rest = params.url.slice(GLYPHS.length + 3);
    const cut = rest.lastIndexOf('/');
    const fontstack = rest.slice(0, cut);
    const range = rest.slice(cut + 1);
    const ours = fontstack.split(',').every((font) => font.startsWith(OFM));
    const template = ours ? openFreeMapGlyphs : projectGlyphs;
    if (!template) throw new Error(`No glyphs for ${fontstack}`);
    const fonts = ours ? fontstack.split(',').map((font) => font.slice(OFM.length)).join(',') : fontstack;
    const response = await fetch(template.replace('{fontstack}', fonts).replace('{range}', range),
        { signal: abortController.signal });
    if (!response.ok) throw new Error(`Glyphs ${fontstack} ${range}: ${response.status}`);
    return { data: await response.arrayBuffer() };
});

const map = new maplibregl.Map({
    container: 'map',
    style,
    // Labels appear at once (MapLibre fades them in over 300 ms): with the
    // previous zoom's labels standing in while tiles load, a fade would
    // only delay them.
    fadeDuration: 0,
    // The map's credits: the background's (each basemap source's), MapLibre's
    // and the plugin's.
    attributionControl: {
        customAttribution: ['<a href="https://maplibre.org/" target="_blank" rel="noopener">MapLibre</a>', CREDIT_HTML],
    },
    center: config.center,
    zoom: config.startZoom ?? config.minZoom,
});
// Without a zoom in the address, open fitted to the export extent (not at
// the lowest zoom, whose few tiles hold all the data at once).
if (config.startZoom === undefined && Array.isArray(config.bounds)) {
    const [west, south, east, north] = config.bounds;
    const camera = map.cameraForBounds([[west, south], [east, north]], { padding: 20 });
    if (camera) map.jumpTo({ center: camera.center, zoom: Math.max(camera.zoom, config.minZoom) });
}
// HTML labels' badges the sprite lacks (new data in a live map), drawn
// from their names; and the empty one of a label part without a badge.
map.on('styleimagemissing', ({ id }) => {
    if (map.hasImage(id)) return;
    const image = drawBadge(id);
    if (image) map.addImage(id, { width: image.width, height: image.height, data: image.data }, { pixelRatio: image.pixelRatio });
});
map.addControl(new maplibregl.NavigationControl(), 'top-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

const popup = new maplibregl.Popup({
    closeButton: false, closeOnClick: false, closeOnMove: false, maxWidth: '280px',
});

// The data layers' ids, worked out once per style change rather than on every
// pointer move (map.getStyle() copies the whole style).
let dataLayerIds = null;
map.on('styledata', () => { dataLayerIds = null; });
const dataLayers = () => {
    dataLayerIds ??= map.getLayersOrder().filter((id) => [SOURCE, LIVE_SOURCE].includes(map.getLayer(id)?.source));
    return dataLayerIds;
};
const firstDataLayer = () => map.getLayersOrder()
    .find((id) => map.getLayer(id)?.type !== 'background' && id !== BASEMAP && !id.startsWith(OFM));

/** An OpenFreeMap icon's name, in its own sprite (added as OFM): every
 * name the property gives, a zoom step's each. */
function openFreeMapImage(value) {
    if (typeof value === 'string') return value ? OFM + value : value;
    if (!Array.isArray(value)) return value;
    if (value[0] === 'step' && JSON.stringify(value[1]) === '["zoom"]') {
        return value.map((item, i) => (i >= 2 && i % 2 === 0 ? openFreeMapImage(item) : item));
    }
    return ['let', 'q2vt_image', ['to-string', value],
        ['case', ['==', ['var', 'q2vt_image'], ''], '', ['concat', OFM, ['var', 'q2vt_image']]]];
}

// The OpenFreeMap layers shown (their style's), and its sources added (by
// id, their definition).
let openFreeMapLayers = [];
const openFreeMapSources = new Map();

/** OpenFreeMap's ``ofm`` style under the data: its layers, sources, fonts
 * and icons marked OFM, apart from the export's. */
function addOpenFreeMap(ofm) {
    openFreeMapGlyphs = ofm.glyphs;
    // The styles share one sprite; another one replaces it.
    const sprite = map.getSprite().find((s) => s.id === 'ofm');
    if (sprite && sprite.url !== ofm.sprite) map.removeSprite('ofm');
    if (ofm.sprite && sprite?.url !== ofm.sprite) map.addSprite('ofm', ofm.sprite);
    // The styles share their sources: kept from one style to the next, with
    // the tiles they have loaded; a visitor's style's own replace them.
    for (const [id, source] of Object.entries(ofm.sources)) {
        const definition = JSON.stringify(source);
        if (openFreeMapSources.get(OFM + id) === definition) continue;
        if (map.getSource(OFM + id)) map.removeSource(OFM + id);
        map.addSource(OFM + id, source);
        openFreeMapSources.set(OFM + id, definition);
    }
    const before = firstDataLayer();
    openFreeMapLayers = [];
    for (const layer of ofm.layers) {
        const copy = structuredClone(layer);
        copy.id = OFM + layer.id;
        if (copy.source) copy.source = OFM + copy.source;
        const layout = copy.layout || {};
        if (Array.isArray(layout['text-font']) && layout['text-font'].every((font) => typeof font === 'string')) {
            layout['text-font'] = layout['text-font'].map((font) => OFM + font);
        }
        if (layout['icon-image'] !== undefined) layout['icon-image'] = openFreeMapImage(layout['icon-image']);
        for (const key of ['fill-pattern', 'line-pattern', 'fill-extrusion-pattern', 'background-pattern']) {
            if (copy.paint?.[key] !== undefined) copy.paint[key] = openFreeMapImage(copy.paint[key]);
        }
        // A layer MapLibre finds wrong (a visitor's style) is left out.
        map.addLayer(copy, before);
        if (map.getLayer(copy.id)) openFreeMapLayers.push(layer);
    }
}

/** The background shown taken away; OpenFreeMap's sources kept when
 * ``keepSources`` (another of its styles comes next). */
function removeBasemap(keepSources = false) {
    if (map.getLayer(BASEMAP)) map.removeLayer(BASEMAP);
    if (map.getSource(BASEMAP)) map.removeSource(BASEMAP);
    for (const layer of openFreeMapLayers) map.removeLayer(OFM + layer.id);
    openFreeMapLayers = [];
    if (keepSources) return;
    for (const id of openFreeMapSources.keys()) map.removeSource(id);
    openFreeMapSources.clear();
}

let shownBasemap = null;
let basemapRequest = 0;

const adapter = {
    ready: new Promise((resolve) => map.once('load', resolve)),
    getZoom: () => map.getZoom(),
    getCenter: () => { const c = map.getCenter(); return [c.lng, c.lat]; },
    onViewChange: (cb) => map.on('move', cb),
    onPointerMove: (cb) => map.on('mousemove', (e) => cb({
        lngLat: [e.lngLat.lng, e.lngLat.lat], point: [e.point.x, e.point.y],
    })),
    onPointerOut: (cb) => map.getCanvasContainer().addEventListener('mouseleave', cb),
    featuresAt: (point) => map.queryRenderedFeatures(point, { layers: dataLayers() })
        .map((f) => withDescription(f.sourceLayer, f.properties)),
    setCursor: (css) => { map.getCanvas().style.cursor = css; },
    showPopup: (lngLat, element) => popup.setLngLat(lngLat).setDOMContent(element).addTo(map),
    hidePopup: () => popup.remove(),
    // A found feature is shown at the closest zoom at which its layer is
    // still drawn: a point at its layer's highest zoom, an area fitted to
    // the view but kept within the layer's zoom levels.
    fitBounds([west, south, east, north], zooms = {}) {
        const max = Math.min(zooms.max ?? 16, map.getMaxZoom());
        const min = Math.min(zooms.min ?? 0, max);
        if (west === east && south === north) {
            map.flyTo({ center: [west, south], zoom: max });
            return;
        }
        const camera = map.cameraForBounds([[west, south], [east, north]], { padding: 60, maxZoom: max });
        const zoom = Math.min(Math.max(camera?.zoom ?? max, min), max);
        map.flyTo({ center: camera?.center ?? [(west + east) / 2, (south + north) / 2], zoom });
    },
    setHighlight(lngLat) {
        const data = {
            type: 'FeatureCollection',
            features: lngLat ? [{ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: lngLat } }] : [],
        };
        if (map.getSource(HIGHLIGHT)) {
            map.getSource(HIGHLIGHT).setData(data);
            return;
        }
        map.addSource(HIGHLIGHT, { type: 'geojson', data });
        map.addLayer({
            id: HIGHLIGHT, type: 'circle', source: HIGHLIGHT,
            paint: {
                'circle-radius': 9, 'circle-color': '#2563eb', 'circle-opacity': 0.25,
                'circle-stroke-color': '#2563eb', 'circle-stroke-width': 3,
            },
        });
    },
    async setBasemap(basemap) {
        const request = ++basemapRequest;
        const ofmStyle = basemap.openfreemap ? basemap.mapStyle : null;
        // OpenFreeMap's 3D tilts the map; leaving it, the map is flat again.
        const wasTilted = shownBasemap?.openfreemap && shownBasemap.mapStyle === '3d';
        if (ofmStyle === '3d' && !wasTilted) map.easeTo({ pitch: 60, duration: 800 });
        if (ofmStyle !== '3d' && wasTilted) map.easeTo({ pitch: 0, duration: 800 });
        // OpenFreeMap already shown in this style: only its labels' language changes.
        if (ofmStyle && shownBasemap?.openfreemap && shownBasemap.mapStyle === ofmStyle
            && shownBasemap.customVersion === basemap.customVersion && openFreeMapLayers.length) {
            for (const layer of openFreeMapLayers) {
                if (layer.layout && showsNames(layer)) map.setLayoutProperty(OFM + layer.id, 'text-field', nameExpression(basemap.language));
            }
            shownBasemap = basemap;
            return;
        }
        removeBasemap(Boolean(ofmStyle));
        shownBasemap = basemap;
        if (ofmStyle) {
            let ofm;
            try {
                ofm = await openFreeMapStyle(ofmStyle);
            } catch (error) {
                console.warn(error);
                return;
            }
            if (request !== basemapRequest) return;
            try {
                addOpenFreeMap(localized(ofm, basemap.language));
            } catch (error) {
                // A visitor's style MapLibre cannot draw: what was added taken away.
                console.warn('Background not shown:', error);
                removeBasemap();
                if (ofmStyle === 'custom') window.alert(`Your style cannot be shown: ${error.message}`);
            }
            return;
        }
        if (!basemap.url) return;
        map.addSource(BASEMAP, {
            type: 'raster', tiles: [basemap.url], tileSize: 256,
            maxzoom: basemap.maxzoom, attribution: basemap.attribution,
        });
        map.addLayer({ id: BASEMAP, type: 'raster', source: BASEMAP }, firstDataLayer());
    },
};

startViewer(adapter, config);

// The live layers' tiles asked again every liveRefresh seconds (their data
// changes): each tile is redrawn when its new one arrives, the old meanwhile.
if (liveUrl && style.sources[LIVE_SOURCE]) {
    refreshLive(liveUrl, Number(new URLSearchParams(window.location.search).get('refresh') ?? config.liveRefresh ?? 30),
        (url) => map.getSource(LIVE_SOURCE)?.setTiles([url]));
}
