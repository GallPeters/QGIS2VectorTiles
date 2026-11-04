// Background maps the viewer can show under the vector tiles. "Project"
// shows the project's background colour (from the style), which also works
// offline. OpenFreeMap is drawn from vector tiles, its labels in a language
// of the visitor's choice (see openfreemap.js).

import { createSegmented, settingsRow } from './controls.js';
import { LANGUAGES, OPENFREEMAP_STYLE } from './openfreemap.js';
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
    {
        id: 'openfreemap',
        name: 'OpenFreeMap',
        title: 'OpenStreetMap as vector tiles, its labels in a language of your choice',
        style: OPENFREEMAP_STYLE,
    },
];

const STORAGE_KEY = 'q2vt:basemap';
const LANGUAGE_KEY = 'q2vt:basemap-language';

/** The background choice, as a segmented control, and OpenFreeMap's label
 * language (its row shown with it); ``apply(basemap)`` shows a background,
 * with its ``language``. The first background shown: the address's
 * ?basemap= (none, osm, bluemarble, openfreemap), else the visitor's last
 * choice, else ``initial`` (the page's default), else the project's
 * background; the language: ?lang= (local, en, ar, he), else the last one,
 * else local. */
export function createBasemapControl({ apply, initial }) {
    const find = (id) => BASEMAPS.find((b) => b.id === id);
    const params = new URLSearchParams(window.location.search);
    let current = find(params.get('basemap')) || find(load(STORAGE_KEY, initial)) || BASEMAPS[0];
    const known = (id) => LANGUAGES.some(([language]) => language === id);
    let language = [params.get('lang'), load(LANGUAGE_KEY, 'local')].find(known) || 'local';
    const show = () => apply({ ...current, language });

    const backgrounds = createSegmented({
        label: 'Background',
        columns: 2,
        options: BASEMAPS.map((b) => ({ id: b.id, label: b.name, title: b.title })),
        current: current.id,
        onChoose: (id) => {
            current = find(id);
            save(STORAGE_KEY, id);
            languageRow.hidden = !current.style;
            show();
        },
    });
    const languages = createSegmented({
        label: 'Labels',
        columns: 2,
        options: LANGUAGES.map(([id, name, title]) => ({ id, label: name, title })),
        current: language,
        onChoose: (id) => {
            language = id;
            save(LANGUAGE_KEY, id);
            show();
        },
    });
    const languageRow = settingsRow('Labels', languages.element);
    languageRow.hidden = !current.style;
    return { rows: [settingsRow('Background', backgrounds.element), languageRow], apply: show };
}
