// HTML labels' parts on a background colour ("badges", see the plugin's
// html_label.py) are images in the label's text. The export draws those in
// its data into the sprite; a badge it lacks (new data in a live map) is
// drawn here from its name, which describes it:
//   q2vtb|<font family and style>|<size in points>|<text colour>|<background>|<text>

export const BADGE_PREFIX = 'q2vtb|';
export const EMPTY_IMAGE = 'q2vt-none';

const STYLE_WEIGHTS = [
    [/black|heavy/i, 900], [/extra ?bold|ultra ?bold/i, 800], [/semi ?bold|demi ?bold/i, 600],
    [/bold/i, 700], [/medium/i, 500], [/extra ?light|ultra ?light/i, 200], [/light/i, 300], [/thin/i, 100],
];
const STYLE_WORDS = /\s+(black|heavy|extra ?bold|ultra ?bold|semi ?bold|demi ?bold|bold|medium|extra ?light|ultra ?light|light|thin|regular|normal|book|italic|oblique)$/i;

/** A font key ("Open Sans SemiBold Italic") as a CSS font at ``px``. */
function cssFont(key, px) {
    let family = key;
    let style = '';
    while (STYLE_WORDS.test(family)) {
        style = family.match(STYLE_WORDS)[1] + ' ' + style;
        family = family.replace(STYLE_WORDS, '');
    }
    const weight = (STYLE_WEIGHTS.find(([pattern]) => pattern.test(style)) || [null, 400])[1];
    const italic = /italic|oblique/i.test(style) ? 'italic ' : '';
    return `${italic}${weight} ${px}px "${family}", sans-serif`;
}

/** The image of a badge name, at the screen's pixel ratio:
 * {width, height, data, pixelRatio}; null for other names. */
export function drawBadge(name) {
    if (name === EMPTY_IMAGE) return { width: 1, height: 1, data: new Uint8Array(4), pixelRatio: 4 };
    if (!name.startsWith(BADGE_PREFIX)) return null;
    const parts = name.slice(BADGE_PREFIX.length).split('|');
    if (parts.length < 5) return null;
    const [font, size, color, background] = parts;
    const text = parts.slice(4).join('|');
    const ratio = Math.max(1, Math.round(window.devicePixelRatio || 1));
    const px = Number(size) * 96 / 72 * ratio;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = cssFont(font, px);
    const metrics = context.measureText(text);
    const ascent = metrics.fontBoundingBoxAscent ?? px * 1.07;
    const descent = metrics.fontBoundingBoxDescent ?? px * 0.29;
    canvas.width = Math.max(1, Math.ceil(metrics.width));
    canvas.height = Math.max(1, Math.ceil(ascent + descent));
    context.fillStyle = background;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.font = cssFont(font, px);
    context.fillStyle = color;
    context.textBaseline = 'alphabetic';
    context.fillText(text, 0, ascent);
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    return { width: canvas.width, height: canvas.height, data: new Uint8Array(data.buffer), pixelRatio: ratio };
}
