// The viewer's controls, shared by the MapLibre and OpenLayers viewers. Each
// viewer passes an adapter with the same interface over its map library:
//
//   ready                      Promise, resolved once the style has loaded
//   getZoom()                  current zoom (MapLibre's numbering)
//   getCenter()                [lon, lat] of the view's centre
//   onViewChange(cb)           cb() whenever the view moves or zooms
//   onPointerMove(cb)          cb({lngLat: [lon, lat], point: [x, y]})
//   onPointerOut(cb)           cb() when the pointer leaves the map
//   featuresAt(point)          properties of the map's features under a pixel
//   setCursor(css)             the map's cursor
//   showPopup(lngLat, el)      show el in a popup at lngLat
//   hidePopup()
//   fitBounds(bbox, zooms)     zoom to [west, south, east, north], within
//                              zooms {min, max} (the style's zoom levels
//                              showing the feature; null if unknown)
//   setHighlight(lngLat|null)  mark a found feature
//   setBasemap(basemap)        show a BASEMAPS entry under the tiles
//   addRightControl(element)   place element under the map's zoom buttons
//   createDrawAdapter()        a Terra Draw adapter for the map
//
// Layout: one column on the left (search and zoom level, the map settings
// panel); on the right the map's zoom buttons with the drawing tools under
// them, as wide as they are.

import { el, icon } from './dom.js';
import { createBasemapControl } from './basemaps.js';
import { createSegmented, createToggle, settingsRow } from './controls.js';
import { createDrawing } from './draw.js';
import { formatCoordinate } from './geo.js';
import { popupContent } from './popup.js';
import { createSearch } from './search.js';
import { load, save } from './storage.js';

const VIEWERS = [['ml', 'MapLibre'], ['ol', 'OpenLayers']];

/** Reopen the page in another map library at the same place and zoom. */
function switchViewer(map, viewer) {
    try { window.localStorage.setItem('q2vt:viewer', viewer); } catch (e) { /* private mode */ }
    const params = new URLSearchParams(window.location.search);
    const [lon, lat] = map.getCenter();
    params.set('viewer', viewer);
    params.set('z', map.getZoom().toFixed(2));
    params.set('c', `${lon.toFixed(6)},${lat.toFixed(6)}`);
    window.location.search = params.toString();
}

const PANEL_KEY = 'q2vt:settings-open';

/** The collapsible "Map settings" card: open by default except on phones,
 * then as the user left it. */
function createSettingsPanel(rows) {
    let open = load(PANEL_KEY, !window.matchMedia('(max-width: 640px)').matches);
    const body = el('div', { class: 'q2vt-panel-body', id: 'q2vt-settings' }, rows);
    const header = el('button', {
        type: 'button', class: 'q2vt-panel-head', 'aria-controls': 'q2vt-settings',
        onclick: () => { open = !open; save(PANEL_KEY, open); render(); },
    }, [el('span', {}, 'Map settings'), icon('chevron')]);
    const panel = el('section', { class: 'q2vt-panel', 'aria-label': 'Map settings' }, [header, body]);
    function render() {
        panel.classList.toggle('open', open);
        header.setAttribute('aria-expanded', String(open));
        body.hidden = !open;
    }
    render();
    return panel;
}

export async function startViewer(map, config) {
    let popupsEnabled = false;
    let overPopup = false;
    let closeTimer = null;

    const zoomLabel = el('div', { class: 'q2vt-zoom', 'aria-live': 'off', title: 'Zoom level' }, 'Zoom –');

    const popupToggle = createToggle({
        label: 'Popups',
        on: popupsEnabled,
        onChange: (on) => {
            popupsEnabled = on;
            if (!on) hidePopup();
        },
    });
    // The other map library reopens the page at the same place and zoom.
    const viewerControl = createSegmented({
        label: 'Viewer',
        options: VIEWERS.map(([id, name]) => ({ id, label: name, title: `Show this map with ${name}` })),
        current: config.viewer,
        onChoose: (id) => switchViewer(map, id),
    });
    const basemaps = createBasemapControl({ apply: (b) => map.setBasemap(b) });
    const settings = createSettingsPanel([
        settingsRow('Popups', popupToggle.element),
        settingsRow('Viewer', viewerControl.element),
        settingsRow('Background', basemaps.element),
    ]);

    const search = createSearch({
        url: config.searchUrl,
        onSelect: (bbox, zooms) => {
            map.fitBounds(bbox, zooms);
            map.setHighlight([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]);
        },
        onClear: () => map.setHighlight(null),
    });
    const drawing = createDrawing({
        createAdapter: () => map.createDrawAdapter(),
        storageKey: config.storageKey,
        onToolChange: (active) => { if (active) hidePopup(); },
    });
    const coords = el('div', { class: 'q2vt-coords', hidden: true });

    document.body.append(
        el('div', { class: 'q2vt-top-left' }, [
            el('div', { class: 'q2vt-search-row' }, [search, zoomLabel]),
            settings,
        ]),
        coords,
    );
    map.addRightControl(drawing.element);

    function hidePopup() {
        clearTimeout(closeTimer);
        overPopup = false;
        map.hidePopup();
    }
    function scheduleClose() {
        clearTimeout(closeTimer);
        closeTimer = setTimeout(() => { if (!overPopup) map.hidePopup(); }, 200);
    }

    await map.ready;

    basemaps.apply();
    drawing.start();

    const updateZoom = () => { zoomLabel.textContent = `Zoom ${map.getZoom().toFixed(2)}`; };
    updateZoom();
    map.onViewChange(updateZoom);

    // Pointer moves come faster than frames: handle the latest once per frame.
    let latestMove = null;
    map.onPointerMove((move) => {
        if (latestMove === null) requestAnimationFrame(() => {
            const pending = latestMove;
            latestMove = null;
            handlePointerMove(pending);
        });
        latestMove = move;
    });
    function handlePointerMove({ lngLat, point }) {
        coords.textContent = formatCoordinate(lngLat);
        coords.hidden = false;
        if (!popupsEnabled || drawing.isActive()) {
            if (!drawing.isActive()) map.setCursor('');
            return;
        }
        if (overPopup) return;
        const features = map.featuresAt(point);
        map.setCursor(features.length ? 'pointer' : '');
        if (!features.length) {
            scheduleClose();
            return;
        }
        clearTimeout(closeTimer);
        const content = popupContent(features[0]);
        content.addEventListener('mouseenter', () => { overPopup = true; clearTimeout(closeTimer); });
        content.addEventListener('mouseleave', () => { overPopup = false; scheduleClose(); });
        map.showPopup(lngLat, content);
    }
    map.onPointerOut(() => {
        coords.hidden = true;
        scheduleClose();
    });
}
