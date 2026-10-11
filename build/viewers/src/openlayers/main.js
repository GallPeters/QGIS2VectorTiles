// OpenLayers viewer: the map library adapter for the shared controls.

import Map from 'ol/Map.js';
import View from 'ol/View.js';
import Feature from 'ol/Feature.js';
import Overlay from 'ol/Overlay.js';
import Point from 'ol/geom/Point.js';
import GeoJSON from 'ol/format/GeoJSON.js';
import TileLayer from 'ol/layer/Tile.js';
import VectorLayer from 'ol/layer/Vector.js';
import VectorSource from 'ol/source/Vector.js';
import XYZ from 'ol/source/XYZ.js';
import Projection from 'ol/proj/Projection.js';
import { fromLonLat, getUserProjection, toLonLat, transformExtent } from 'ol/proj.js';
import { Circle, Fill, Icon, Stroke, Style } from 'ol/style.js';
import { defaults as defaultControls } from 'ol/control/defaults.js';
import ScaleLine from 'ol/control/ScaleLine.js';
import { apply } from 'ol-mapbox-style';
import { TerraDrawOpenLayersAdapter } from 'terra-draw-openlayers-adapter';
import { startViewer } from '../shared/app.js';
import '../shared/ui.css';

const SOURCE = 'q2vt_tiles';
const config = window.Q2VT_CONFIG;

const map = new Map({
    target: 'map',
    view: new View({ center: fromLonLat(config.center), zoom: config.minZoom }),
    controls: defaultControls({ attributionOptions: { collapsible: true } })
        .extend([new ScaleLine({ units: 'metric' })]),
});

const popupElement = document.createElement('div');
popupElement.className = 'q2vt-ol-popup';
const popup = new Overlay({
    element: popupElement, positioning: 'bottom-center', stopEvent: true, offset: [0, -10],
});
map.addOverlay(popup);

// Below Terra Draw's layer (zIndex 900), whose canvas takes the drawing clicks.
const highlight = new VectorLayer({
    source: new VectorSource(),
    zIndex: 800,
    style: new Style({
        image: new Circle({
            radius: 9,
            fill: new Fill({ color: 'rgba(37, 99, 235, 0.25)' }),
            stroke: new Stroke({ color: '#2563eb', width: 3 }),
        }),
    }),
});

let basemapLayer = null;
const isDataLayer = (layer) => layer.get('mapbox-source') === SOURCE;

const ready = apply(map, config.styleUrl).then(() => {
    // The style may constrain the view; allow every zoom, as before.
    const old = map.getView();
    map.setView(new View({
        center: fromLonLat(config.center),
        zoom: config.minZoom ?? old.getZoom(),
        minZoom: 0,
        maxZoom: 22,
        constrainResolution: false,
        showFullExtent: true,
    }));
    map.addLayer(highlight);
    // Terra Draw listens on the map's top canvas, which exists once the
    // map has rendered.
    return new Promise((resolve) => map.once('rendercomplete', resolve));
});

const adapter = {
    ready,
    getZoom: () => map.getView().getZoom(),
    onViewChange: (cb) => {
        map.getView().on('change:resolution', cb);
        map.on('moveend', cb);
    },
    onPointerMove: (cb) => map.on('pointermove', (e) => {
        if (e.dragging) return;
        cb({ lngLat: toLonLat(e.coordinate), point: e.pixel });
    }),
    onPointerOut: (cb) => map.getViewport().addEventListener('mouseleave', cb),
    featuresAt(point) {
        const found = [];
        map.forEachFeatureAtPixel(point, (feature) => {
            found.push(feature.getProperties());
            return true;  // The topmost feature only.
        }, { hitTolerance: 2, layerFilter: isDataLayer });
        return found;
    },
    setCursor: (css) => { map.getViewport().style.cursor = css; },
    showPopup(lngLat, element) {
        popupElement.replaceChildren(element);
        popup.setPosition(fromLonLat(lngLat));
    },
    hidePopup: () => popup.setPosition(undefined),
    fitBounds([west, south, east, north]) {
        const view = map.getView();
        if (west === east && south === north) {
            view.animate({ center: fromLonLat([west, south]), zoom: Math.max(view.getZoom(), 16), duration: 500 });
        } else {
            view.fit(transformExtent([west, south, east, north], 'EPSG:4326', 'EPSG:3857'),
                { padding: [60, 60, 60, 60], maxZoom: 17, duration: 500 });
        }
    },
    setHighlight(lngLat) {
        const source = highlight.getSource();
        source.clear();
        if (lngLat) source.addFeature(new Feature(new Point(fromLonLat(lngLat))));
    },
    setBasemap(basemap) {
        if (basemapLayer) map.removeLayer(basemapLayer);
        basemapLayer = null;
        if (!basemap.url) return;
        basemapLayer = new TileLayer({
            source: new XYZ({
                url: basemap.url, maxZoom: basemap.maxzoom,
                attributions: basemap.attribution, crossOrigin: 'anonymous',
            }),
        });
        // Above the style's background colour, below the tiles.
        const layers = map.getLayers();
        const background = layers.getArray().findIndex((l) => (l.get('mapbox-layers') || []).includes('background'));
        layers.insertAt(background + 1, basemapLayer);
    },
    createDrawAdapter() {
        const drawAdapter = new TerraDrawOpenLayersAdapter({
            map,
            zIndex: 900,
            lib: {
                Fill, Feature, GeoJSON, Style, Circle, VectorLayer, VectorSource, Stroke, Icon,
                Projection, getUserProjection, fromLonLat, toLonLat,
            },
        });
        // The adapter listens on the map's top canvas, but OpenLayers
        // replaces its canvases when layers change (e.g. a new basemap).
        // It listens on the viewport instead, which viewer.html makes the
        // pointer events' target (Terra Draw ignores events aimed elsewhere).
        drawAdapter.getMapEventElement = () => map.getViewport();
        return drawAdapter;
    },
};

startViewer(adapter, config);
