// Feature popups shown on hover: the plugin's q2vt_* metadata first, then
// the feature's own attributes.

import { escapeHtml } from './dom.js';

// Properties the map libraries add to features, and the plugin's own
// bookkeeping (which rule of a shared dataset a feature is drawn by); not
// user data.
const LIBRARY_PROPERTIES = new Set(['geometry', 'layer', 'mvt:layer', 'q2vt_rule']);

// "{key: value, key: value}" strings, as the plugin writes descriptions.
function parsePseudoDict(text) {
    const trimmed = text.trim();
    if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return null;
    const result = {};
    for (const pair of trimmed.slice(1, -1).split(',')) {
        const parts = pair.split(':');
        if (parts.length < 2) continue;
        result[parts[0].trim()] = parts.slice(1).join(':').trim();
    }
    return result;
}

function renderObject(obj) {
    const rows = Object.entries(obj).map(([key, value]) => `
        <tr><th>${escapeHtml(key)}</th><td>${
            typeof value === 'object' && value !== null ? renderObject(value) : escapeHtml(value)
        }</td></tr>`);
    return `<table class="q2vt-kv">${rows.join('')}</table>`;
}

function formatValue(value) {
    if (value === null || value === undefined) return '<i class="q2vt-null">null</i>';
    if (typeof value === 'string') {
        const parsed = parsePseudoDict(value);
        return parsed ? renderObject(parsed) : escapeHtml(value);
    }
    if (typeof value === 'object') return renderObject(value);
    return escapeHtml(value);
}

/** The popup's content element for a feature's properties. */
export function popupContent(properties) {
    const metadata = [];
    const fields = [];
    for (const [key, value] of Object.entries(properties || {})) {
        if (LIBRARY_PROPERTIES.has(key)) continue;
        if (key.startsWith('q2vt_')) metadata.push([key.slice(5), value]);
        else fields.push([key, value]);
    }
    const item = ([key, value], cls) =>
        `<div class="${cls}"><div class="q2vt-key">${escapeHtml(key)}</div>`
        + `<div class="q2vt-value">${formatValue(value)}</div></div>`;
    const div = document.createElement('div');
    div.className = 'q2vt-popup';
    div.innerHTML = (metadata.length
        ? `<div class="q2vt-meta"><div class="q2vt-meta-title">metadata</div>${
            metadata.map((m) => item(m, 'q2vt-meta-item')).join('')}</div>`
        : '') + fields.map((f) => item(f, 'q2vt-field')).join('');
    return div;
}
