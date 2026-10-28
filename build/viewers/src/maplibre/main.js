// MapLibre viewer: the map library adapter for the shared controls.
// maplibre-gl.mjs is not bundled; it sits next to bundle.js and loads its
// worker (maplibre-gl-worker.mjs) from the same folder.

import * as maplibregl from './maplibre-gl.mjs';
import { Protocol } from 'pmtiles';
import { TerraDrawMapLibreGLAdapter } from 'terra-draw-maplibre-gl-adapter';
import { startViewer } from '../shared/app.js';
import { archives, loadStyle, tileUrls, withDescription } from '../shared/style.js';
import '../shared/ui.css';

const SOURCE = 'q2vt_tiles';
const BASEMAP = 'q2vt-basemap';
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

const map = new maplibregl.Map({
    container: 'map',
    style,
    // Labels shown at once, as QGIS draws them: MapLibre's fade (300 ms by
    // default, sometimes twice after a move) was most of the wait after
    // each pan or zoom.
    fadeDuration: 0,
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
    .find((id) => map.getLayer(id)?.type !== 'background' && id !== BASEMAP);

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
    setBasemap(basemap) {
        if (map.getLayer(BASEMAP)) map.removeLayer(BASEMAP);
        if (map.getSource(BASEMAP)) map.removeSource(BASEMAP);
        if (!basemap.url) return;
        map.addSource(BASEMAP, {
            type: 'raster', tiles: [basemap.url], tileSize: 256,
            maxzoom: basemap.maxzoom, attribution: basemap.attribution,
        });
        map.addLayer({ id: BASEMAP, type: 'raster', source: BASEMAP }, firstDataLayer());
    },
    // MapLibre stacks the controls of a corner in the order they are added.
    addRightControl(element) {
        element.classList.add('maplibregl-ctrl');
        map.addControl({ onAdd: () => element, onRemove: () => element.remove() }, 'top-right');
    },
    createDrawAdapter: () => new TerraDrawMapLibreGLAdapter({ map }),
};

startViewer(adapter, config);
