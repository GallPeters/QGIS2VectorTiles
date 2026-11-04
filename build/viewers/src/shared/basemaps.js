// Background maps the viewer can show under the vector tiles. "Project"
// shows the project's background colour (from the style), which also works
// offline. OpenFreeMap is drawn from vector tiles, in one of its styles,
// its labels in a language of the visitor's choice (see openfreemap.js).

import { createSegmented, settingsRow } from './controls.js';
import { el } from './dom.js';
import { checkStyle, CUSTOM, DEFAULT_STYLE, hasCustomStyle, LANGUAGES, setCustomStyle, STYLES } from './openfreemap.js';
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
        title: 'OpenStreetMap as vector tiles, in a style and label language of your choice',
        openfreemap: true,
    },
];

const STORAGE_KEY = 'q2vt:basemap';
const STYLE_KEY = 'q2vt:basemap-style';
const LANGUAGE_KEY = 'q2vt:basemap-language';
const CUSTOM_KEY = 'q2vt:basemap-custom';

/** The first of ``choices`` that is one of ``options`` ([id, ...] each). */
const firstKnown = (options, ...choices) => choices.find((id) => options.some(([known]) => known === id));

/** The background choice, as a segmented control, and OpenFreeMap's style
 * and label language (their rows shown with it); ``apply(basemap)`` shows a
 * background, with its ``mapStyle`` and ``language``. The first background
 * shown: the address's ?basemap= (none, osm, bluemarble, openfreemap), else
 * the visitor's last choice, else ``initial`` (the page's default), else the
 * project's background; the style: ?mapstyle= (positron, bright, liberty,
 * dark, fiord, 3d), else the last one, else Liberty; the language: ?lang=
 * (local, en, ar, he), else the last one, else local.
 *
 * "Upload style" shows a style.json the visitor chooses (or drops on the
 * page): kept in their browser for their next visits, seen by them only. */
export function createBasemapControl({ apply, initial }) {
    const find = (id) => BASEMAPS.find((b) => b.id === id);
    const params = new URLSearchParams(window.location.search);
    let current = find(params.get('basemap')) || find(load(STORAGE_KEY, initial)) || BASEMAPS[0];
    const stored = load(CUSTOM_KEY, null);
    try {
        if (stored) setCustomStyle(checkStyle(stored));
    } catch (error) {
        console.warn('Your style not shown:', error);
    }
    let mapStyle = firstKnown(STYLES, params.get('mapstyle'), load(STYLE_KEY, DEFAULT_STYLE)) || DEFAULT_STYLE;
    if (mapStyle === CUSTOM && !hasCustomStyle()) mapStyle = DEFAULT_STYLE;
    let language = firstKnown(LANGUAGES, params.get('lang'), load(LANGUAGE_KEY, 'local')) || 'local';
    // Counts the styles of theirs the visitor chose: another one is drawn
    // anew, though its id (CUSTOM) is the same.
    let customVersion = 0;
    const show = () => apply({ ...current, mapStyle, language, customVersion });

    /** A style.json file of the visitor's shown, OpenFreeMap chosen with it;
     * one that is not a style is said so, the background left as it was. */
    async function useFile(file) {
        let style;
        try {
            style = checkStyle(JSON.parse(await file.text()));
        } catch (error) {
            window.alert(`${file.name} cannot be shown: ${error instanceof SyntaxError ? 'not a JSON file' : error.message}.`);
            return;
        }
        setCustomStyle(style);
        // Too large for the browser's storage: shown for this visit only.
        if (!save(CUSTOM_KEY, style)) console.warn('Your style is kept for this visit only (too large to store).');
        customVersion += 1;
        mapStyle = CUSTOM;
        save(STYLE_KEY, CUSTOM);
        styles.set(CUSTOM);
        if (!current.openfreemap) {
            current = find('openfreemap');
            save(STORAGE_KEY, current.id);
            backgrounds.set(current.id);
            showRows();
        }
        show();
    }
    const picker = el('input', { type: 'file', accept: '.json,application/json', hidden: true });
    picker.addEventListener('change', () => {
        if (picker.files[0]) useFile(picker.files[0]);
        picker.value = '';  // The same file can be chosen again.
    });
    const choose = () => picker.click();
    // A style.json dropped on the page.
    window.addEventListener('dragover', (event) => {
        if ([...event.dataTransfer.items].some((item) => item.kind === 'file')) event.preventDefault();
    });
    window.addEventListener('drop', (event) => {
        if (!event.dataTransfer.files.length) return;
        event.preventDefault();  // Not opened by the browser in place of the map.
        const file = [...event.dataTransfer.files].find((f) => /\.json$/i.test(f.name) || f.type === 'application/json');
        if (file) useFile(file);
    });

    const backgrounds = createSegmented({
        label: 'Background',
        columns: 2,
        options: BASEMAPS.map((b) => ({ id: b.id, label: b.name, title: b.title })),
        current: current.id,
        onChoose: (id) => {
            current = find(id);
            save(STORAGE_KEY, id);
            showRows();
            show();
        },
    });
    const styles = createSegmented({
        label: 'Style',
        columns: 2,
        options: STYLES.map(([id, name, title]) => ({
            // Across the row, under the six styles.
            id, label: name, title, wide: id === CUSTOM, onAgain: id === CUSTOM ? choose : undefined,
        })),
        current: mapStyle,
        onChoose: (id) => {
            // None of theirs yet: one is chosen first (the style shown till then).
            if (id === CUSTOM && !hasCustomStyle()) {
                styles.set(mapStyle);
                choose();
                return;
            }
            mapStyle = id;
            save(STYLE_KEY, id);
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
    // OpenFreeMap's own choices, shown with it only.
    const ofmRows = [settingsRow('Style', styles.element), settingsRow('Labels', languages.element)];
    const showRows = () => ofmRows.forEach((row) => { row.hidden = !current.openfreemap; });
    showRows();
    return { rows: [settingsRow('Background', backgrounds.element), ...ofmRows, picker], apply: show };
}
