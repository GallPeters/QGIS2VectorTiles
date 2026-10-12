// The viewer's controls, shared by the MapLibre and OpenLayers viewers. Each
// viewer passes an adapter with the same interface over its map library:
//
//   ready                      Promise, resolved once the style has loaded
//   getZoom()                  current zoom
//   onViewChange(cb)           cb() whenever the view moves or zooms
//   onPointerMove(cb)          cb({lngLat: [lon, lat], point: [x, y]})
//   onPointerOut(cb)           cb() when the pointer leaves the map
//   featuresAt(point)          properties of the map's features under a pixel
//   setCursor(css)             the map's cursor
//   showPopup(lngLat, el)      show el in a popup at lngLat
//   hidePopup()
//   fitBounds(bbox)            zoom to [west, south, east, north]
//   setHighlight(lngLat|null)  mark a found feature
//   setBasemap(basemap)        show a BASEMAPS entry under the tiles
//   createDrawAdapter()        a Terra Draw adapter for the map

import { el } from './dom.js';
import { createBasemapSwitcher } from './basemaps.js';
import { createDrawing } from './draw.js';
import { formatCoordinate } from './geo.js';
import { popupContent } from './popup.js';
import { createSearch } from './search.js';

export async function startViewer(map, config) {
    let popupsEnabled = false;
    let overPopup = false;
    let closeTimer = null;

    const zoomChip = el('div', { class: 'q2vt-chip', 'aria-live': 'off' }, 'Zoom: –');
    const popupToggle = el('button', {
        type: 'button', class: 'q2vt-chip q2vt-chip-button', 'aria-pressed': 'false',
        onclick: () => {
            popupsEnabled = !popupsEnabled;
            renderPopupToggle();
            if (!popupsEnabled) hidePopup();
        },
    });
    function renderPopupToggle() {
        popupToggle.textContent = popupsEnabled ? 'Popups: ON' : 'Popups: OFF';
        popupToggle.setAttribute('aria-pressed', String(popupsEnabled));
        popupToggle.classList.toggle('active', popupsEnabled);
    }
    renderPopupToggle();

    const search = createSearch({
        url: config.searchUrl,
        onSelect: (bbox) => {
            map.fitBounds(bbox);
            map.setHighlight([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]);
        },
        onClear: () => map.setHighlight(null),
    });
    const drawing = createDrawing({
        createAdapter: () => map.createDrawAdapter(),
        storageKey: config.storageKey,
        onToolChange: (active) => { if (active) hidePopup(); },
    });
    const basemaps = createBasemapSwitcher({ apply: (b) => map.setBasemap(b) });
    const coords = el('div', { class: 'q2vt-coords', hidden: true });

    document.body.append(
        el('div', { class: 'q2vt-top-left' }, [search, el('div', { class: 'q2vt-row' }, [zoomChip, popupToggle])]),
        drawing.element,
        basemaps.element,
        coords,
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
    drawing.start();

    const updateZoom = () => { zoomChip.textContent = `Zoom: ${map.getZoom().toFixed(2)}`; };
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
