import Map from 'ol/Map.js';
import View from 'ol/View.js';
import TileLayer from 'ol/layer/Tile.js';
import OSM from 'ol/source/OSM.js';
import Overlay from 'ol/Overlay.js';
import { fromLonLat } from 'ol/proj.js';
import { defaults as defaultControls } from 'ol/control/defaults.js';
import { apply } from 'ol-mapbox-style';

/* ===================================================== */
/* STATE                                                 */
/* ===================================================== */
let popupsEnabled = true;
let rasterEnabled = false;
let isOverPopup   = false;
let closeTimeout  = null;

/* ===================================================== */
/* OPTIONAL OSM RASTER BACKGROUND                        */
/* ===================================================== */
const osmLayer = new TileLayer({
    source: new OSM(),
    visible: rasterEnabled,
    zIndex: -1
});

/* ===================================================== */
/* MAP                                                   */
/* ===================================================== */
const map = new Map({
    target: 'map',
    layers: [osmLayer],
    view: new View({
        center: fromLonLat([0, 20]),
        zoom: 2,
        constrainResolution: false
    }),
    controls: defaultControls({
        attributionOptions: { collapsible: true }
    })
});

/* ===================================================== */
/* APPLY MAPLIBRE/MAPBOX STYLE                           */
/* ===================================================== */
apply(map, 'http://localhost:18111991/style/style.json')
    .then(() => {
        const old = map.getView();

        map.setView(new View({
            center: old.getCenter(),
            zoom:   old.getZoom(),
            minZoom: 0,
            maxZoom: 22,
            constrainResolution: false,
            showFullExtent: true
        }));

        // Re-bind the zoom indicator to the NEW view
        map.getView().on('change:resolution', updateZoomIndicator);

        try {
            map.getView().setZoom(_Q2VT_MINZOOM);
            map.getView().setCenter(fromLonLat(_Q2VT_CENTER));
        } catch (e) {
            console.warn('Initial view placeholders not substituted:', e);
        }

        updateZoomIndicator();
    })
    .catch((err) => {
        console.error('Failed to apply style:', err);
    }); 

/* ===================================================== */
/* ZOOM INDICATOR                                        */
/* ===================================================== */
function updateZoomIndicator() {
    const z = map.getView().getZoom();
    if (z === undefined || z === null) return;
    document.getElementById('zoom-indicator').textContent =
        'Zoom: ' + z.toFixed(2);
}
map.getView().on('change:resolution', updateZoomIndicator);
map.on('moveend', updateZoomIndicator);
updateZoomIndicator();

/* ===================================================== */
/* POPUP OVERLAY                                         */
/* ===================================================== */
const popupEl      = document.getElementById('popup');
const popupContent = document.getElementById('popup-content');

const popupOverlay = new Overlay({
    element: popupEl,
    autoPan: false,
    stopEvent: true,
    positioning: 'bottom-center',
    offset: [0, 0]
});
map.addOverlay(popupOverlay);

popupEl.addEventListener('mouseenter', () => {
    isOverPopup = true;
    clearTimeout(closeTimeout);
});
popupEl.addEventListener('mouseleave', () => {
    isOverPopup = false;
    scheduleClose();
});

function showPopup(coord, html) {
    popupContent.innerHTML = html;
    popupEl.classList.add('visible');
    popupOverlay.setPosition(coord);
}

function hidePopup() {
    popupEl.classList.remove('visible');
    popupOverlay.setPosition(undefined);
}

function scheduleClose() {
    clearTimeout(closeTimeout);
    closeTimeout = setTimeout(() => {
        if (!isOverPopup) hidePopup();
    }, 200);
}

/* ===================================================== */
/* POPUP TOGGLE BUTTON                                   */
/* ===================================================== */
document.getElementById('popup-toggle').addEventListener('click', () => {
    popupsEnabled = !popupsEnabled;

    const btn = document.getElementById('popup-toggle');
    btn.textContent = popupsEnabled ? 'Popups: ON' : 'Popups: OFF';

    if (!popupsEnabled) {
        hidePopup();
    }
});



/* ===================================================== */
/* HTML ESCAPE                                           */
/* ===================================================== */
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/* ===================================================== */
/* PSEUDO-DICT PARSER                                    */
/* ===================================================== */
function parsePseudoDict(str) {
    str = str.trim();
    if (!str.startsWith('{') || !str.endsWith('}')) return null;

    str = str.slice(1, -1);
    const result = {};

    for (const pair of str.split(',')) {
        const parts = pair.split(':');
        if (parts.length < 2) continue;

        const key   = parts[0].trim();
        const value = parts.slice(1).join(':').trim();

        result[key] = value;
    }

    return result;
}

/* ===================================================== */
/* RECURSIVE OBJECT RENDERER                             */
/* ===================================================== */
function renderObject(obj) {
    let html = `
        <table style="
            border-collapse: collapse;
            font-size: 11px;
            width: 100%;
            background: transparent;
        ">
    `;

    for (const [key, value] of Object.entries(obj)) {
        html += `
            <tr>
                <td style="
                    font-weight: 700;
                    padding: 3px 6px;
                    border-bottom: 1px solid #eeeeee;
                    color: #111827;
                    white-space: nowrap;
                ">
                    ${escapeHtml(key)}
                </td>
                <td style="
                    padding: 3px 6px;
                    border-bottom: 1px solid #eeeeee;
                    color: #374151;
                ">
                    ${typeof value === 'object' && value !== null
                        ? renderObject(value)
                        : escapeHtml(value)}
                </td>
            </tr>
        `;
    }

    html += '</table>';
    return html;
}

/* ===================================================== */
/* VALUE FORMATTER                                       */
/* ===================================================== */
function formatValue(value) {
    if (value === null || value === undefined) {
        return '<i style="color:#9ca3af;">null</i>';
    }

    if (typeof value === 'string') {
        const parsed = parsePseudoDict(value);
        if (parsed) return renderObject(parsed);
        return escapeHtml(value);
    }

    if (typeof value === 'object') {
        return renderObject(value);
    }

    return escapeHtml(value);
}

/* ===================================================== */
/* BUILD POPUP HTML                                      */
/* ===================================================== */
function buildPopupHtml(feature) {
    const rawProps = (typeof feature.getProperties === 'function')
        ? feature.getProperties()
        : (feature.properties || {});

    const props = {};
    for (const k in rawProps) {
        if (k === 'geometry') continue;     
        if (k === 'layer')    continue;   // OL/MVT source-layer name; not user data
        props[k] = rawProps[k];
    }

    const q2vtFields   = [];
    const normalFields = [];

    for (const [key, value] of Object.entries(props)) {
        if (key.startsWith('q2vt_')) {
            q2vtFields.push([key.replace('q2vt_', ''), value]);
        } else {
            normalFields.push([key, value]);
        }
    }

    let html = `
        <div style="
            font-family: Arial, sans-serif;
            font-size: 12px;
            max-width: 240px;
            max-height: 300px;
            overflow-y: auto;
            padding: 6px;
            line-height: 1.25;
            background: #ffffff;
            color: #111827;
            border-radius: 6px;
        ">
    `;

    if (q2vtFields.length > 0) {
        html += `
            <div style="
                margin-bottom: 8px;
                padding: 6px;
                background: #f8fafc;
                border: 1px solid #eeeeee;
                border-radius: 6px;
            ">
                <div style="
                    font-weight: 800;
                    font-size: 11px;
                    color: #2e7d32;
                    margin-bottom: 6px;
                ">
                    metadata
                </div>
        `;

        for (const [key, value] of q2vtFields) {
            html += `
                <div style="margin-bottom: 5px;">
                    <div style="
                        font-weight: 700;
                        color: #1e88e5;
                        font-size: 11px;
                        margin-bottom: 2px;
                    ">
                        ${escapeHtml(key)}
                    </div>
                    <div style="color:#374151; font-size:12px;">
                        ${formatValue(value)}
                    </div>
                </div>
            `;
        }

        html += `</div>`;
    }

    for (const [key, value] of normalFields) {
        html += `
            <div style="margin-bottom: 6px;">
                <div style="
                    font-weight: 800;
                    color: #111827;
                    font-size: 11px;
                    margin-bottom: 2px;
                ">
                    ${escapeHtml(key)}
                </div>
                <div style="color:#374151; font-size:12px;">
                    ${formatValue(value)}
                </div>
            </div>
        `;
    }

    html += `</div>`;
    return html;
}

/* ===================================================== */
/* HOVER FEATURE INSPECTION                              */
/* ===================================================== */
map.on('pointermove', (evt) => {
    if (evt.dragging) return;
    if (!popupsEnabled) return;
    if (isOverPopup) return;

    let pickedFeature = null;
    map.forEachFeatureAtPixel(
        evt.pixel,
        (feature) => {
            if (!pickedFeature) {
                pickedFeature = feature;
                return true;
            }
            return false;
        },
        {
            hitTolerance: 2,
            layerFilter: (lyr) => lyr !== osmLayer
        }
    );

    if (!pickedFeature) {
        scheduleClose();
        return;
    }

    const html = buildPopupHtml(pickedFeature);
    showPopup(evt.coordinate, html);
});

map.getViewport().addEventListener('mouseleave', () => {
    scheduleClose();
});

/* cursor feedback */
map.on('pointermove', (evt) => {
    if (evt.dragging) return;
    const hit = map.hasFeatureAtPixel(evt.pixel, {
        hitTolerance: 2,
        layerFilter: (lyr) => lyr !== osmLayer
    });
    map.getViewport().style.cursor = hit && popupsEnabled ? 'pointer' : '';
});

