// Label priority in OpenLayers.
//
// MapLibre places labels layer by layer (the style's last layer first),
// and within a layer by symbol-sort-key, which the style sets from the QGIS
// label priority. OpenLayers declutters by style z-index, highest first;
// ol-mapbox-style sets it to the layer's position in the style, so the
// layers' order already holds, but not the order within a layer. Here each
// label's z-index also gets its sort key, as a fraction below the next
// layer's position.

// The expressions the plugin writes sort keys with (see
// maplibre_converter.get_text_sort_key, merge_rule_layers and
// merge_zoom_bands); anything else gives NaN, and the label keeps its
// layer's z-index.
function evaluate(expression, properties, zoom) {
    if (typeof expression === 'number') return expression;
    if (!Array.isArray(expression)) return expression ?? null;
    const [op, ...args] = expression;
    const value = (e) => evaluate(e, properties, zoom);
    const number = (e) => Number(value(e));
    switch (op) {
    case 'literal': return args[0];
    case 'get': return properties[args[0]] ?? null;
    case 'zoom': return zoom;
    case '-': return args.length === 1 ? -number(args[0]) : number(args[0]) - number(args[1]);
    case '+': return args.reduce((sum, e) => sum + number(e), 0);
    case '*': return args.reduce((product, e) => product * number(e), 1);
    case 'min': return Math.min(...args.map(number));
    case 'max': return Math.max(...args.map(number));
    case 'coalesce':
        for (const e of args) {
            const v = value(e);
            if (v !== null && v !== undefined) return v;
        }
        return null;
    case 'to-number':
        for (const e of args) {
            const v = value(e);
            if (v === null || v === undefined || v === '') continue;
            const n = Number(v);
            if (Number.isFinite(n)) return n;
        }
        return 0;
    case 'match': {
        const input = value(args[0]);
        for (let i = 1; i + 1 < args.length; i += 2) {
            const labels = Array.isArray(args[i]) ? args[i] : [args[i]];
            if (labels.includes(input)) return value(args[i + 1]);
        }
        return value(args[args.length - 1]);
    }
    case 'step': {
        const input = number(args[0]);
        let result = value(args[1]);
        for (let i = 2; i + 1 < args.length; i += 2) {
            if (input < args[i]) break;
            result = value(args[i + 1]);
        }
        return result;
    }
    default: return NaN;
    }
}

/**
 * Give the labels of the style's layers with a per-feature sort key a
 * z-index in their sort key order, within their layer's.
 * @param {import('ol/layer/VectorTile.js').default} olLayer A layer of the style.
 * @param {Object} style The style (MapLibre JSON) applied to it.
 * @param {(resolution: number) => number} zoomOf The style's zoom at a resolution.
 */
export function applySortKeys(olLayer, style, zoomOf) {
    const keys = {};
    style.layers.forEach((layer, index) => {
        const key = layer.type === 'symbol' && layer.layout?.['symbol-sort-key'];
        if (Array.isArray(key)) keys[index] = key;
    });
    if (!Object.keys(keys).length) return;
    const styleFunction = olLayer.getStyleFunction();
    if (!styleFunction) return;
    olLayer.setStyle((feature, resolution, onlyLayer) => {
        const styles = styleFunction(feature, resolution, onlyLayer);
        if (!Array.isArray(styles)) return styles;
        let properties;
        for (const olStyle of styles) {
            const index = Math.floor(olStyle.getZIndex() ?? NaN);
            const key = keys[index];
            if (!key || (!olStyle.getText() && !olStyle.getImage())) continue;
            properties ??= feature.getProperties();
            // Rounded to tenths: OpenLayers draws each z-index as a group,
            // so as many groups as there are priorities, not labels.
            const sortKey = Math.round(Number(evaluate(key, properties, zoomOf(resolution))) * 10) / 10;
            // Lower keys are placed first: a larger fraction, in (0, 1).
            if (Number.isFinite(sortKey)) olStyle.setZIndex(index + 0.5 - Math.atan(sortKey) / Math.PI);
        }
        return styles;
    });
}
