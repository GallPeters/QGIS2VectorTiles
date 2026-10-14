// Background maps the viewer can show under the vector tiles. "Project"
// shows the project's background colour (from the style), which also works offline.

import { createSegmented } from './controls.js';
import { load, save } from './storage.js';

export const BASEMAPS = [
    { id: 'none', name: 'Project', title: "The project's background colour (works offline)", url: null },
    {
        id: 'osm',
        name: 'OSM',
        title: 'OpenStreetMap',
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        maxzoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
    },
    {
        id: 'bluemarble',
        name: 'Blue Marble',
        title: 'NASA Blue Marble',
        url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg',
        maxzoom: 8,
        attribution: 'Imagery: <a href="https://earthdata.nasa.gov/gibs" target="_blank">NASA EOSDIS GIBS</a>',
    },
];

const STORAGE_KEY = 'q2vt:basemap';

/** The basemap choice, as a segmented control; ``apply(basemap)`` shows it. */
export function createBasemapControl({ apply }) {
    let current = BASEMAPS.find((b) => b.id === load(STORAGE_KEY, 'none')) || BASEMAPS[0];
    const control = createSegmented({
        label: 'Background',
        options: BASEMAPS.map((b) => ({ id: b.id, label: b.name, title: b.title })),
        current: current.id,
        onChoose: (id) => {
            current = BASEMAPS.find((b) => b.id === id);
            save(STORAGE_KEY, id);
            apply(current);
        },
    });
    return { element: control.element, apply: () => apply(current) };
}
