// Drawing and measuring, with Terra Draw (the same tools on MapLibre and
// OpenLayers). Drawings are kept in the browser (per map) and can be
// downloaded as GeoJSON; measurements are temporary, one at a time.

import {
    TerraDraw,
    TerraDrawCircleMode,
    TerraDrawLineStringMode,
    TerraDrawPointMode,
    TerraDrawPolygonMode,
    TerraDrawRectangleMode,
    TerraDrawSelectMode,
} from 'terra-draw';
import { el, icon } from './dom.js';
import { describeGeometry, formatArea, formatLength, lineLength, polygonArea } from './geo.js';
import { load, save } from './storage.js';

const DRAW = '#2563eb';
const MEASURE = '#ea580c';
const WHITE = '#ffffff';
const MEASURE_LINE = 'measure-line';
const MEASURE_AREA = 'measure-polygon';

const TOOLS = [
    { mode: 'select', icon: 'select', title: 'Select and edit drawings',
      hint: 'Click a drawing to select it; drag it or its points to edit. Press Delete to remove it.' },
    { mode: 'point', icon: 'point', title: 'Draw points', hint: 'Click to add a point.' },
    { mode: 'linestring', icon: 'line', title: 'Draw a line',
      hint: 'Click to add points; click the last point again to finish.' },
    { mode: 'polygon', icon: 'polygon', title: 'Draw a polygon',
      hint: 'Click to add corners; click the first corner to close it.' },
    { mode: 'rectangle', icon: 'rectangle', title: 'Draw a rectangle', hint: 'Click two opposite corners.' },
    { mode: 'circle', icon: 'circle', title: 'Draw a circle', hint: 'Click the centre, then click to set the radius.' },
    { separator: true },
    { mode: MEASURE_LINE, icon: 'ruler', title: 'Measure distance',
      hint: 'Click to add points; click the last point again to finish.' },
    { mode: MEASURE_AREA, icon: 'area', title: 'Measure area',
      hint: 'Click to add corners; click the first corner to close it.' },
];

const polygonStyle = (color) => ({
    fillColor: color, fillOpacity: 0.15, outlineColor: color, outlineWidth: 2,
    closingPointColor: WHITE, closingPointOutlineColor: color, closingPointWidth: 5, closingPointOutlineWidth: 2,
});
const lineStyle = (color) => ({
    lineStringColor: color, lineStringWidth: 3,
    closingPointColor: WHITE, closingPointOutlineColor: color, closingPointWidth: 5, closingPointOutlineWidth: 2,
});
const shapeStyle = (color) => ({
    fillColor: color, fillOpacity: 0.15, outlineColor: color, outlineWidth: 2,
});

const editable = { draggable: true, coordinates: { midpoints: true, draggable: true, deletable: true } };

function createModes() {
    return [
        new TerraDrawSelectMode({
            flags: {
                point: { feature: { draggable: true } },
                linestring: { feature: editable },
                polygon: { feature: editable },
                rectangle: { feature: { draggable: true, coordinates: { resizable: 'opposite' } } },
                circle: { feature: { draggable: true, coordinates: { resizable: 'center-fixed' } } },
            },
            styles: {
                selectedPolygonColor: DRAW, selectedPolygonFillOpacity: 0.3,
                selectedPolygonOutlineColor: '#1d4ed8', selectedPolygonOutlineWidth: 3,
                selectedLineStringColor: '#1d4ed8', selectedLineStringWidth: 4,
                selectedPointColor: '#1d4ed8', selectedPointWidth: 7,
                selectedPointOutlineColor: WHITE, selectedPointOutlineWidth: 2,
                selectionPointColor: WHITE, selectionPointOutlineColor: '#1d4ed8',
                selectionPointWidth: 5, selectionPointOutlineWidth: 2,
                midPointColor: WHITE, midPointOutlineColor: DRAW, midPointWidth: 3, midPointOutlineWidth: 1,
            },
        }),
        new TerraDrawPointMode({
            styles: { pointColor: DRAW, pointWidth: 6, pointOutlineColor: WHITE, pointOutlineWidth: 2 },
        }),
        new TerraDrawLineStringMode({ styles: lineStyle(DRAW) }),
        new TerraDrawPolygonMode({ styles: polygonStyle(DRAW) }),
        new TerraDrawRectangleMode({ styles: shapeStyle(DRAW) }),
        new TerraDrawCircleMode({ styles: shapeStyle(DRAW) }),
        new TerraDrawLineStringMode({ modeName: MEASURE_LINE, styles: lineStyle(MEASURE) }),
        new TerraDrawPolygonMode({ modeName: MEASURE_AREA, styles: polygonStyle(MEASURE) }),
    ];
}

// Points Terra Draw adds while drawing and editing (closing, coordinate,
// selection, mid and snapping points); they carry their feature's mode.
const GUIDES = ['selectionPoint', 'midPoint', 'closingPoint', 'snappingPoint', 'coordinatePoint'];
const isGuide = (feature) => GUIDES.some((key) => feature.properties[key]);
const isMeasureMode = (feature) => String(feature.properties.mode).startsWith('measure-');
const isMeasure = (feature) => feature && isMeasureMode(feature) && !isGuide(feature)
    && (feature.geometry.type === 'LineString' || feature.geometry.type === 'Polygon');

function measurementText(feature) {
    const g = feature.geometry;
    if (g.type === 'LineString') {
        return `Distance: <b>${formatLength(lineLength(g.coordinates))}</b>`;
    }
    if (g.type === 'Polygon') {
        return `Area: <b>${formatArea(polygonArea(g.coordinates))}</b><br>`
            + `Perimeter: ${formatLength(lineLength(g.coordinates[0]))}`;
    }
    return '';
}

/**
 * The drawing panel. ``createAdapter()`` returns the map library's Terra
 * Draw adapter; ``storageKey`` identifies this map's saved drawings.
 * ``start()`` must be called once the map has loaded.
 */
export function createDrawing({ createAdapter, storageKey, onToolChange }) {
    const key = `q2vt:drawings:${storageKey}`;
    let draw = null;
    let tool = null;          // Active mode name, or null when panning the map.
    let selectedId = null;
    let measureId = null;     // The current measurement's feature.
    let saveTimer = null;

    const buttons = new Map();
    const toolbar = el('div', { class: 'q2vt-draw-tools', hidden: true });
    for (const t of TOOLS) {
        if (t.separator) {
            toolbar.append(el('div', { class: 'q2vt-sep' }));
            continue;
        }
        const button = el('button', {
            type: 'button', class: 'q2vt-tool', title: t.title, 'aria-label': t.title,
            onclick: () => setTool(tool === t.mode ? null : t.mode),
        }, icon(t.icon));
        buttons.set(t.mode, button);
        toolbar.append(button);
    }
    toolbar.append(
        el('div', { class: 'q2vt-sep' }),
        el('button', {
            type: 'button', class: 'q2vt-tool', title: 'Delete the selected drawing',
            'aria-label': 'Delete the selected drawing', onclick: deleteSelected,
        }, icon('trash')),
        el('button', {
            type: 'button', class: 'q2vt-tool', title: 'Delete all drawings',
            'aria-label': 'Delete all drawings', onclick: clearAll,
        }, icon('clear')),
        el('button', {
            type: 'button', class: 'q2vt-tool', title: 'Download drawings (GeoJSON)',
            'aria-label': 'Download drawings as GeoJSON', onclick: download,
        }, icon('download')),
    );
    const toggle = el('button', {
        type: 'button', class: 'q2vt-tool q2vt-draw-toggle', title: 'Draw and measure',
        'aria-label': 'Draw and measure', 'aria-expanded': 'false',
        onclick: () => setOpen(toolbar.hidden),
    }, icon('pencil'));
    const readout = el('div', { class: 'q2vt-readout', hidden: true });
    const panel = el('div', { class: 'q2vt-draw' }, [toggle, toolbar, readout]);

    function setOpen(open) {
        toolbar.hidden = !open;
        toggle.classList.toggle('active', open);
        toggle.setAttribute('aria-expanded', String(open));
        if (!open) setTool(null);
    }

    function removeMeasurement() {
        if (draw && measureId !== null && draw.hasFeature(measureId)) draw.removeFeatures([measureId]);
        measureId = null;
    }

    function setTool(mode) {
        if (!draw) return;
        if (!String(mode).startsWith('measure-')) removeMeasurement();
        tool = mode;
        selectedId = null;
        draw.setMode(mode || 'static');
        for (const [m, b] of buttons) b.classList.toggle('active', m === mode);
        updateReadout();
        onToolChange(!!mode);
    }

    function updateReadout() {
        if (!tool) {
            readout.hidden = true;
            return;
        }
        const t = TOOLS.find((x) => x.mode === tool);
        let html = `<div class="q2vt-hint">${t.hint}</div>`;
        const measured = measureId !== null && draw.hasFeature(measureId) && draw.getSnapshotFeature(measureId);
        if (tool.startsWith('measure-') && measured) {
            html += `<div class="q2vt-measure">${measurementText(measured)}</div>`;
        } else if (tool === 'select' && selectedId !== null && draw.hasFeature(selectedId)) {
            html += `<div class="q2vt-measure">${describeGeometry(draw.getSnapshotFeature(selectedId).geometry)}</div>`;
        }
        readout.innerHTML = html;
        readout.hidden = false;
    }

    function drawings() {
        return draw.getSnapshot().filter((f) => !isMeasureMode(f) && !isGuide(f) && !f.properties.currentlyDrawing);
    }

    function scheduleSave() {
        clearTimeout(saveTimer);
        saveTimer = setTimeout(() => save(key, drawings()), 300);
    }

    function deleteSelected() {
        if (draw && selectedId !== null && draw.hasFeature(selectedId)) {
            draw.removeFeatures([selectedId]);
            selectedId = null;
            updateReadout();
        }
    }

    function clearAll() {
        if (!draw || !drawings().length) return;
        if (!window.confirm('Delete all drawings on this map?')) return;
        draw.removeFeatures(drawings().map((f) => f.id));
        selectedId = null;
        updateReadout();
    }

    function download() {
        if (!draw) return;
        const features = drawings().map((f) => ({
            type: 'Feature', geometry: f.geometry, properties: { shape: f.properties.mode },
        }));
        if (!features.length) {
            window.alert('There are no drawings to download yet.');
            return;
        }
        const blob = new Blob([JSON.stringify({ type: 'FeatureCollection', features }, null, 2)],
            { type: 'application/geo+json' });
        const link = el('a', { href: URL.createObjectURL(blob), download: 'drawings.geojson' });
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }

    function start() {
        draw = new TerraDraw({ adapter: createAdapter(), modes: createModes() });
        draw.on('ready', () => {
            const saved = load(key, []);
            if (Array.isArray(saved) && saved.length) draw.addFeatures(saved);
        });
        draw.on('change', (ids, type) => {
            if (type === 'create') {
                for (const id of ids) {
                    const f = draw.hasFeature(id) && draw.getSnapshotFeature(id);
                    if (isMeasure(f) && id !== measureId) {
                        // A new measurement replaces the previous one, once
                        // Terra Draw has finished creating it.
                        const previous = measureId;
                        measureId = id;
                        if (previous !== null) {
                            setTimeout(() => { if (draw.hasFeature(previous)) draw.removeFeatures([previous]); });
                        }
                    }
                }
            }
            updateReadout();
            scheduleSave();
        });
        draw.on('select', (id) => { selectedId = id; updateReadout(); });
        draw.on('deselect', () => { selectedId = null; updateReadout(); });
        draw.start();
        draw.setMode('static');
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && tool && document.activeElement?.tagName !== 'INPUT') setTool(null);
        });
    }

    return { element: panel, start, isActive: () => tool !== null };
}
