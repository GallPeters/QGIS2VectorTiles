// Small DOM helpers shared by the viewer's controls.

export function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'html') node.innerHTML = value;
        else if (key === 'text') node.textContent = value;
        else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of [].concat(children)) {
        if (child === null || child === undefined || child === false) continue;
        node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
}

export function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

const ICONS = {
    search: '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    pencil: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13 7 4 4"/>',
    select: '<path d="M6 3l12 9-5.5 1.2L15 19l-2.5 1-2.6-5.7L6 18z"/>',
    point: '<circle cx="12" cy="12" r="3.5" fill="currentColor"/><circle cx="12" cy="12" r="7.5"/>',
    line: '<path d="M4 18l5-8 5 4 6-9"/><circle cx="4" cy="18" r="1.6" fill="currentColor"/><circle cx="20" cy="5" r="1.6" fill="currentColor"/>',
    polygon: '<path d="M5 18 4 8l8-4 8 5-3 10z"/>',
    rectangle: '<rect x="4" y="6" width="16" height="12" rx="1"/>',
    circle: '<circle cx="12" cy="12" r="8"/>',
    ruler: '<path d="M3 16 16 3l5 5L8 21z"/><path d="m7 12 2 2M10 9l2 2M13 6l2 2"/>',
    area: '<path d="M4 4h16v16H4z" stroke-dasharray="3 2"/><path d="M8 16 16 8M8 11l3-3M13 16l3-3"/>',
    trash: '<path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/>',
    clear: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="m9 9 6 6M15 9l-6 6"/>',
    download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    layers: '<path d="m12 4 9 5-9 5-9-5z"/><path d="m3 14 9 5 9-5"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
};

export function icon(name) {
    const span = document.createElement('span');
    span.className = 'q2vt-icon';
    span.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" `
        + `stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
    return span;
}
