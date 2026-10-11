// MapLibre viewer: the map library adapter for the shared controls.
// maplibre-gl.mjs is not bundled; it sits next to bundle.js and loads its
// worker (maplibre-gl-worker.mjs) from the same folder.

import * as maplibregl from './maplibre-gl.mjs';
import { TerraDrawMapLibreGLAdapter } from 'terra-draw-maplibre-gl-adapter';
import { startViewer } from '../shared/app.js';
import '../shared/ui.css';

const SOURCE = 'q2vt_tiles';
const BASEMAP = 'q2vt-basemap';
const HIGHLIGHT = 'q2vt-search-hit';
const config = window.Q2VT_CONFIG;

const map = new maplibregl.Map({
    container: 'map',
    style: config.styleUrl,
    center: config.center,
    zoom: config.minZoom,
});
map.addControl(new maplibregl.NavigationControl(), 'top-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

const popup = new maplibregl.Popup({
    closeButton: false, closeOnClick: false, closeOnMove: false, maxWidth: '280px',
});

const dataLayers = () => map.getStyle().layers.filter((l) => l.source === SOURCE).map((l) => l.id);
const firstDataLayer = () => map.getStyle().layers.find((l) => l.type !== 'background' && l.id !== BASEMAP)?.id;

const adapter = {
    ready: new Promise((resolve) => map.once('load', resolve)),
    getZoom: () => map.getZoom(),
    onViewChange: (cb) => map.on('move', cb),
    onPointerMove: (cb) => map.on('mousemove', (e) => cb({
        lngLat: [e.lngLat.lng, e.lngLat.lat], point: [e.point.x, e.point.y],
    })),
    onPointerOut: (cb) => map.getCanvasContainer().addEventListener('mouseleave', cb),
    featuresAt: (point) => map.queryRenderedFeatures(point, { layers: dataLayers() }).map((f) => f.properties),
    setCursor: (css) => { map.getCanvas().style.cursor = css; },
    showPopup: (lngLat, element) => popup.setLngLat(lngLat).setDOMContent(element).addTo(map),
    hidePopup: () => popup.remove(),
    fitBounds([west, south, east, north]) {
        if (west === east && south === north) {
            map.flyTo({ center: [west, south], zoom: Math.max(map.getZoom(), 15) });
        } else {
            map.fitBounds([[west, south], [east, north]], { padding: 60, maxZoom: 16 });
        }
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
    createDrawAdapter: () => new TerraDrawMapLibreGLAdapter({ map }),
};

startViewer(adapter, config);
