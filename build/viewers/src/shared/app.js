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
//
// Layout: one column on the left (search and zoom level, the map settings
// panel); on the right the map's zoom buttons.

import { el, icon } from './dom.js';
import { createBasemapControl } from './basemaps.js';
import { createSegmented, createToggle, settingsRow } from './controls.js';
import { formatCoordinate } from './geo.js';
import { popupContent } from './popup.js';
import { createSearch } from './search.js';
import { load, save } from './storage.js';

// MapLibre draws everything the export makes (balloon signs, stretched
// label backgrounds, labels moving around their point): recommended.
const VIEWERS = [['ml', 'MapLibre', true], ['ol', 'OpenLayers', false]];

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

// What a feature's popup shows: nothing (no popups), its own fields, or the
// plugin's metadata (for checking an export).
const POPUP_MODES = [
    ['off', 'Off', 'No popups'],
    ['fields', 'Fields', "The feature's fields (those exported: all or visible fields)"],
    ['debug', 'Debug', "The plugin's metadata: layer, rule, label text..."],
];

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

/** "Map only": every control hidden but the map's credits, full screen where
 * the browser allows it (phones: no browser bars either); a small button,
 * or leaving full screen, brings them back. */
function createMapOnly() {
    let fullScreened = false;
    const root = document.documentElement;
    const toggle = createToggle({ label: 'Map only', onChange: (on) => set(on) });
    toggle.element.title = 'Hide the controls: only the map (the button in the corner brings them back)';
    const exit = el('button', {
        type: 'button', class: 'q2vt-exit-map-only', title: 'Show the controls',
        'aria-label': 'Show the controls', onclick: () => set(false),
    }, icon('menu'));
    function set(on) {
        root.classList.toggle('q2vt-map-only', on);
        toggle.set(on);
        if (on && !document.fullscreenElement && root.requestFullscreen) {
            root.requestFullscreen().then(() => { fullScreened = true; }).catch(() => {});
        } else if (!on && fullScreened && document.fullscreenElement) {
            fullScreened = false;
            document.exitFullscreen().catch(() => {});
        }
    }
    document.addEventListener('fullscreenchange', () => {
        if (!document.fullscreenElement && fullScreened) {
            fullScreened = false;
            set(false);
        }
    });
    return { toggle: toggle.element, exit };
}

export async function startViewer(map, config) {
    let popupMode = 'off';
    let overPopup = false;
    let closeTimer = null;

    const zoomLabel = el('div', { class: 'q2vt-zoom', 'aria-live': 'off', title: 'Zoom level' }, 'Zoom –');

    const popupControl = createSegmented({
        label: 'Popups',
        columns: 3,
        options: POPUP_MODES.map(([id, label, title]) => ({ id, label, title })),
        current: popupMode,
        onChoose: (id) => {
            popupMode = id;
            hidePopup();  // The next hover shows the new kind.
        },
    });
    // The other map library reopens the page at the same place and zoom.
    const viewerControl = createSegmented({
        label: 'Viewer',
        columns: 2,
        options: VIEWERS.map(([id, name, recommended]) => ({
            id,
            label: name,
            badge: recommended ? 'Recommended' : undefined,
            title: `Show this map with ${name}${recommended ? ' (recommended: draws everything the export makes)' : ''}`,
        })),
        current: config.viewer,
        onChoose: (id) => switchViewer(map, id),
    });
    const basemaps = createBasemapControl({ apply: (b) => map.setBasemap(b), initial: config.basemap });
    const mapOnly = createMapOnly();
    const settings = createSettingsPanel([
        // The switch, then the choices.
        settingsRow('Map only', mapOnly.toggle),
        settingsRow('Popups', popupControl.element),
        settingsRow('Viewer', viewerControl.element),
        ...basemaps.rows,  // Background, and OpenFreeMap's label language.
    ]);

    const search = createSearch({
        url: config.searchUrl,
        onSelect: (bbox, zooms) => {
            map.fitBounds(bbox, zooms);
            map.setHighlight([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]);
        },
        onClear: () => map.setHighlight(null),
    });
    const coords = el('div', { class: 'q2vt-coords', hidden: true });

    document.body.append(
        el('div', { class: 'q2vt-top-left' }, [
            el('div', { class: 'q2vt-search-row' }, [search, zoomLabel]),
            settings,
        ]),
        coords,
        mapOnly.exit,
    );

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
        if (popupMode === 'off') {
            map.setCursor('');
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
        const content = popupContent(features[0], popupMode);
        content.addEventListener('mouseenter', () => { overPopup = true; clearTimeout(closeTimer); });
        content.addEventListener('mouseleave', () => { overPopup = false; scheduleClose(); });
        map.showPopup(lngLat, content);
    }
    map.onPointerOut(() => {
        coords.hidden = true;
        scheduleClose();
    });
}
