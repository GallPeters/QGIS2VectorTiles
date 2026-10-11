// Background maps the viewer can show under the vector tiles. "None" shows
// the project's background colour (from the style), which also works offline.

import { el, icon } from './dom.js';
import { load, save } from './storage.js';

export const BASEMAPS = [
    {
        id: 'osm',
        name: 'OpenStreetMap',
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        maxzoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
    },
    {
        id: 'bluemarble',
        name: 'NASA Blue Marble',
        url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg',
        maxzoom: 8,
        attribution: 'Imagery: <a href="https://earthdata.nasa.gov/gibs" target="_blank">NASA EOSDIS GIBS</a>',
    },
    { id: 'none', name: 'Project background', url: null },
];

const STORAGE_KEY = 'q2vt:basemap';

/** The basemap switcher; ``apply(basemap)`` shows the chosen one on the map. */
export function createBasemapSwitcher({ apply }) {
    let current = BASEMAPS.find((b) => b.id === load(STORAGE_KEY, 'osm')) || BASEMAPS[0];

    const list = el('div', { class: 'q2vt-basemap-list', role: 'menu', hidden: true });
    const button = el('button', {
        type: 'button', class: 'q2vt-tool', title: 'Basemap', 'aria-label': 'Choose the basemap',
        'aria-haspopup': 'menu', onclick: () => { list.hidden = !list.hidden; },
    }, icon('layers'));
    const box = el('div', { class: 'q2vt-basemap' }, [button, list]);

    function render() {
        list.replaceChildren(...BASEMAPS.map((b) => el('button', {
            type: 'button', role: 'menuitemradio', 'aria-checked': String(b === current),
            class: `q2vt-basemap-item${b === current ? ' active' : ''}`,
            onclick: () => choose(b),
        }, b.name)));
    }

    function choose(basemap) {
        current = basemap;
        save(STORAGE_KEY, basemap.id);
        list.hidden = true;
        render();
        apply(basemap);
    }

    document.addEventListener('click', (e) => { if (!box.contains(e.target)) list.hidden = true; });
    render();
    return { element: box, apply: () => apply(current) };
}
