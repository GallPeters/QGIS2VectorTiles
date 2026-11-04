// OpenFreeMap: OpenStreetMap as vector tiles (https://openfreemap.org), drawn
// in the browser from one of its styles, so its labels can be shown in
// another language: each place has its local name ("name") and, where
// OpenStreetMap has them, its names in other languages ("name:en", ...).

/** The styles offered: [id, name shown, title]. 3D is Liberty (whose
 * buildings rise) seen tilted, in MapLibre; flat in OpenLayers. */
export const STYLES = [
    ['positron', 'Positron', 'Light and quiet'],
    ['bright', 'Bright', 'Colourful, detailed'],
    ['liberty', 'Liberty', 'Detailed, soft colours'],
    ['dark', 'Dark', 'Dark and quiet'],
    ['fiord', 'Fiord', 'Dark blue'],
    ['3d', '3D', 'Liberty, tilted, its buildings in 3D (MapLibre)'],
];
export const DEFAULT_STYLE = 'liberty';

/** The address of a style's style.json. */
export const styleUrl = (id) => `https://tiles.openfreemap.org/styles/${id === '3d' ? 'liberty' : id}`;

/** The label languages offered: [id, name shown, title]. */
export const LANGUAGES = [
    ['local', 'Local', "Each place's own name"],
    ['en', 'English', 'English names, else local'],
    ['ar', 'عربي', 'Arabic names, else local'],
    ['he', 'עברית', 'Hebrew names, else local'],
];

const pending = new Map();

/** An OpenFreeMap style, by id (each fetched once). */
export function openFreeMapStyle(id) {
    const url = styleUrl(id);
    if (!pending.has(url)) {
        pending.set(url, fetch(url).then((response) => {
            if (!response.ok) throw new Error(`Cannot load OpenFreeMap (${response.status})`);
            return response.json();
        }).catch((error) => {
            pending.delete(url);  // Tried again next time.
            throw error;
        }));
    }
    return pending.get(url);
}

/** A label's text in ``language``: its name in that language, else its local name. */
export function nameExpression(language) {
    if (language === 'en') return ['coalesce', ['get', 'name:en'], ['get', 'name_en'], ['get', 'name']];
    if (language === 'ar' || language === 'he') return ['coalesce', ['get', `name:${language}`], ['get', 'name']];
    return ['coalesce', ['get', 'name'], ['get', 'name:latin']];
}

/** Whether a layer's labels are place names (not, e.g., road numbers). */
export function showsNames(layer) {
    const field = layer.layout?.['text-field'];
    return field !== undefined && /"name[:_"]/.test(JSON.stringify(field));
}

/** A copy of ``style`` with its place names in ``language``. */
export function localized(style, language) {
    const copy = structuredClone(style);
    for (const layer of copy.layers) {
        if (showsNames(layer)) layer.layout['text-field'] = nameExpression(language);
    }
    return copy;
}
