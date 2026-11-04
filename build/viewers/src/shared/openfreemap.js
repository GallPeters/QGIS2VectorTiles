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
    ['custom', 'Your style…', 'Your own style.json (an OpenFreeMap style edited in Maputnik, say); click again for another'],
];
/** The visitor's own style (a style.json they chose, see setCustomStyle). */
export const CUSTOM = 'custom';
export const DEFAULT_STYLE = 'liberty';

/** The address of a style's style.json. */
export const styleUrl = (id) => `https://tiles.openfreemap.org/styles/${id === '3d' || id === 'custom' ? 'liberty' : id}`;

/** The label languages offered: [id, name shown, title]. */
export const LANGUAGES = [
    ['local', 'Local', "Each place's own name"],
    ['en', 'English', 'English names, else local'],
    ['ar', 'عربي', 'Arabic names, else local'],
    ['he', 'עברית', 'Hebrew names, else local'],
];

const pending = new Map();
let customStyle = null;
let customPending = null;

/** Whether ``style`` reads as a MapLibre style (version 8, its layers and
 * sources); throws, saying why, if not. */
export function checkStyle(style) {
    if (!style || typeof style !== 'object' || Array.isArray(style)) throw new Error('not a style.json');
    if (style.version !== 8) throw new Error('not a MapLibre style (version 8)');
    if (!Array.isArray(style.layers) || !style.layers.length) throw new Error('the style has no layers');
    if (!style.sources || typeof style.sources !== 'object') throw new Error('the style has no sources');
    return style;
}

/** The visitor's own style, shown when OpenFreeMap's style is CUSTOM. */
export function setCustomStyle(style) {
    customStyle = style;
    customPending = null;
}
export const hasCustomStyle = () => customStyle !== null;

// A full address (not relative, nor mapbox://), which the style can be read from.
const absolute = (url) => typeof url === 'string' && /^(https?:\/\/|data:)/.test(url);

/** The visitor's style, what it lacks taken from Liberty: a file saved
 * from an editor may have no fonts or icons, or give them (or its tiles)
 * by addresses relative to where it came from, which a file chosen in the
 * browser has not. */
async function completed(style) {
    const copy = structuredClone(style);
    const sprite = Array.isArray(copy.sprite) ? copy.sprite.find((s) => absolute(s?.url))?.url : copy.sprite;
    const tiles = (source) => source.type !== 'vector' && source.type !== 'raster'
        || absolute(source.url) || (Array.isArray(source.tiles) && source.tiles.every(absolute));
    const lacking = !absolute(copy.glyphs) || !absolute(sprite)
        || Object.values(copy.sources).some((source) => !tiles(source));
    copy.sprite = absolute(sprite) ? sprite : undefined;
    if (!lacking) return copy;
    const base = await openFreeMapStyle(DEFAULT_STYLE);
    if (!absolute(copy.glyphs)) copy.glyphs = base.glyphs;
    if (!copy.sprite) copy.sprite = base.sprite;
    for (const [id, source] of Object.entries(copy.sources)) {
        if (!tiles(source)) copy.sources[id] = base.sources[id] || Object.values(base.sources)[0];
    }
    return copy;
}

/** An OpenFreeMap style, by id (each fetched once); CUSTOM, the visitor's. */
export function openFreeMapStyle(id) {
    if (id === CUSTOM) {
        if (!customStyle) return Promise.reject(new Error('No style of yours chosen'));
        customPending ??= completed(customStyle);
        return customPending;
    }
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
        if (layer.layout && showsNames(layer)) layer.layout['text-field'] = nameExpression(language);
    }
    return copy;
}
