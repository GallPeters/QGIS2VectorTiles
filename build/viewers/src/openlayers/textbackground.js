// Label background shapes in OpenLayers.
//
// MapLibre stretches a label's background icon around its text
// (icon-text-fit); ol-mapbox-style can't, and would draw the icon at its
// small sprite size. The style gives such layers their box in metadata
// ("q2vt:text-background": fill, stroke, stroke-width, padding), which
// OpenLayers draws as the text's own background instead of the icon: a
// rectangle (rounded corners and ellipses are MapLibre's only).

import Fill from 'ol/style/Fill.js';
import Stroke from 'ol/style/Stroke.js';

/**
 * Draw the background boxes of the style's labels that have one.
 * @param {import('ol/layer/VectorTile.js').default} olLayer A layer of the style.
 * @param {Object} style The style (MapLibre JSON) applied to it.
 */
export function applyTextBackgrounds(olLayer, style) {
    const boxes = {};
    style.layers.forEach((layer, index) => {
        const box = layer.metadata?.['q2vt:text-background'];
        if (box) {
            boxes[index] = {
                fill: new Fill({ color: box.fill }),
                stroke: box['stroke-width'] > 0
                    ? new Stroke({ color: box.stroke, width: box['stroke-width'] }) : undefined,
                padding: box.padding,
            };
        }
    });
    if (!Object.keys(boxes).length) return;
    const styleFunction = olLayer.getStyleFunction();
    if (!styleFunction) return;
    olLayer.setStyle((feature, resolution, onlyLayer) => {
        const styles = styleFunction(feature, resolution, onlyLayer);
        if (!Array.isArray(styles)) return styles;
        for (const olStyle of styles) {
            const box = boxes[Math.floor(olStyle.getZIndex() ?? NaN)];
            const text = olStyle.getText();
            if (!box || !text) continue;
            text.setBackgroundFill(box.fill);
            text.setBackgroundStroke(box.stroke);
            text.setPadding(box.padding);
            olStyle.setImage(undefined);
        }
        return styles;
    });
}
