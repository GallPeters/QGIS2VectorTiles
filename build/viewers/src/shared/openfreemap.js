// OpenFreeMap: OpenStreetMap as vector tiles (https://openfreemap.org), drawn
// in the browser from its Liberty style, so its labels can be shown in
// another language: each place has its local name ("name") and, where
// OpenStreetMap has them, its names in other languages ("name:en", ...).

export const OPENFREEMAP_STYLE = 'https://tiles.openfreemap.org/styles/liberty';

/** The label languages offered: [id, name shown, title]. */
export const LANGUAGES = [
    ['local', 'Local', "Each place's own name"],
    ['en', 'English', 'English names, else local'],
    ['ar', 'عربي', 'Arabic names, else local'],
    ['he', 'עברית', 'Hebrew names, else local'],
];

let pending = null;

/** The OpenFreeMap style (fetched once). */
export function openFreeMapStyle() {
    pending ??= fetch(OPENFREEMAP_STYLE).then((response) => {
        if (!response.ok) throw new Error(`Cannot load OpenFreeMap (${response.status})`);
        return response.json();
    }).catch((error) => {
        pending = null;  // Tried again next time.
        throw error;
    });
    return pending;
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
