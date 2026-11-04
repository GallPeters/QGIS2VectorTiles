// MapLibre viewer: the map library adapter for the shared controls.
// maplibre-gl.mjs is not bundled; it sits next to bundle.js and loads its
// worker (maplibre-gl-worker.mjs) from the same folder.

import * as maplibregl from './maplibre-gl.mjs';
import { Protocol } from 'pmtiles';
import { startViewer } from '../shared/app.js';
import { CREDIT_HTML } from '../shared/credit.js';
import { localized, nameExpression, openFreeMapStyle, showsNames } from '../shared/openfreemap.js';
import { archives, loadStyle, tileUrls, withDescription } from '../shared/style.js';
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
const style = await loadStyle(config.styleUrl, tileUrls.maplibre);
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
    dataLayerIds ??= map.getLayersOrder().filter((id) => map.getLayer(id)?.source === SOURCE);
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

// The OpenFreeMap layers shown (their style's), and its sources added.
let openFreeMapLayers = [];
let openFreeMapSources = [];

/** OpenFreeMap's ``ofm`` style under the data: its layers, sources, fonts
 * and icons marked OFM, apart from the export's. */
function addOpenFreeMap(ofm) {
    openFreeMapGlyphs = ofm.glyphs;
    if (ofm.sprite && !map.getSprite().some((sprite) => sprite.id === 'ofm')) map.addSprite('ofm', ofm.sprite);
    for (const [id, source] of Object.entries(ofm.sources)) {
        map.addSource(OFM + id, source);
        openFreeMapSources.push(OFM + id);
    }
    const before = firstDataLayer();
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
        map.addLayer(copy, before);
    }
    openFreeMapLayers = ofm.layers;
}

function removeBasemap() {
    if (map.getLayer(BASEMAP)) map.removeLayer(BASEMAP);
    if (map.getSource(BASEMAP)) map.removeSource(BASEMAP);
    for (const layer of openFreeMapLayers) map.removeLayer(OFM + layer.id);
    for (const id of openFreeMapSources) map.removeSource(id);
    openFreeMapLayers = [];
    openFreeMapSources = [];
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
        // OpenFreeMap already shown, in another language: only its labels change.
        if (basemap.style && shownBasemap?.id === basemap.id && openFreeMapLayers.length) {
            for (const layer of openFreeMapLayers) {
                if (showsNames(layer)) map.setLayoutProperty(OFM + layer.id, 'text-field', nameExpression(basemap.language));
            }
            shownBasemap = basemap;
            return;
        }
        removeBasemap();
        shownBasemap = basemap;
        if (basemap.style) {
            let ofm;
            try {
                ofm = await openFreeMapStyle();
            } catch (error) {
                console.warn(error);
                return;
            }
            if (request === basemapRequest) addOpenFreeMap(localized(ofm, basemap.language));
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
