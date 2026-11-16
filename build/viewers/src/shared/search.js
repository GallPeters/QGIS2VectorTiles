// Search across every exported feature's fields, from the index the plugin
// writes at export time (style/search.json), version 2:
//   {"version": 2, "strings": [each distinct text once],
//    "layers": [{"name": ..., "fields": [...], "minzoom", "maxzoom",
//                "text": [1 where a field's values are indices into strings],
//                "rows": [[values...], ...], "boxes": [w, s, e, n, ...]}],
//    "truncated": bool}
// Version 1 ("features": [[layer index, [values...], [w, s, e, n]], ...]) is
// read too.
// The index is loaded the first time the search box is used.

import { el, escapeHtml, icon } from './dom.js';

const MAX_RESULTS = 50;

/** Version 2 indexes (each text once in "strings", rows and boxes per
 * layer) as the features list the search reads; texts are shared, not
 * copied. Version 1 indexes are read as they are. */
function unpack(data) {
    if (data.version !== 2) return { layers: data.layers || [], features: data.features || [] };
    const strings = data.strings || [];
    const features = [];
    const layers = (data.layers || []).map((layer, index) => {
        const text = layer.text || [];
        (layer.rows || []).forEach((row, r) => {
            const values = row.map((v, i) => (text[i] && typeof v === 'number' ? strings[v] : v));
            features.push([index, values, layer.boxes.slice(4 * r, 4 * r + 4)]);
        });
        const { rows: _rows, boxes: _boxes, text: _text, ...rest } = layer;
        return rest;
    });
    return { layers, features, strings };
}

class SearchIndex {
    constructor(data) {
        const { layers, features, strings } = unpack(data);
        this.layers = layers;
        this.features = features;
        this.truncated = !!data.truncated;
        // Lower-cased values, per feature, built once (each distinct text once).
        const lowered = new Map((strings || []).map((s) => [s, s.toLowerCase()]));
        const lower = (v) => {
            if (v === null || v === undefined) return '';
            const text = String(v);
            let low = lowered.get(text);
            if (low === undefined) { low = text.toLowerCase(); lowered.set(text, low); }
            return low;
        };
        this.lower = this.features.map(([, values]) => values.map(lower));
    }

    /** Features whose fields contain every word of the query, best first. */
    find(query) {
        const q = query.trim().toLowerCase();
        if (!q) return [];
        const words = q.split(/\s+/);
        const results = [];
        for (let i = 0; i < this.features.length; i++) {
            const values = this.lower[i];
            if (!words.every((w) => values.some((v) => v.includes(w)))) continue;
            // The field that best matches the whole query names the result.
            let best = -1;
            let score = 0;
            values.forEach((v, j) => {
                const s = v === q ? 3 : v.startsWith(q) ? 2 : v.includes(q) ? 1 : 0;
                if (s > score || (s === score && best >= 0 && s > 0 && v.length < values[best].length)) {
                    score = s;
                    best = j;
                }
            });
            if (best < 0) best = values.findIndex((v) => v.includes(words[0]));
            results.push({ index: i, field: best, score });
        }
        results.sort((a, b) => b.score - a.score
            || this.lower[a.index][a.field].length - this.lower[b.index][b.field].length);
        return results;
    }

    describe({ index, field }) {
        const [layerIndex, values, bbox] = this.features[index];
        const layer = this.layers[layerIndex] || { name: '', fields: [] };
        return {
            value: String(values[field]), layer: layer.name, field: layer.fields[field], bbox,
            zooms: { min: layer.minzoom ?? null, max: layer.maxzoom ?? null },
        };
    }
}

function highlight(text, query) {
    const escaped = escapeHtml(text);
    const words = query.trim().split(/\s+/).filter(Boolean)
        .map((w) => escapeHtml(w).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    if (!words.length) return escaped;
    return escaped.replace(new RegExp(`(${words.join('|')})`, 'gi'), '<mark>$1</mark>');
}

/**
 * The search box. ``onSelect(bbox, zooms)`` is called with the chosen
 * feature's [west, south, east, north] and the style zoom levels {min, max}
 * its layer is drawn at; ``onClear()`` when the search is cleared.
 */
export function createSearch({ url, onSelect, onClear }) {
    let index = null;
    let loading = null;
    let results = [];
    let active = -1;

    const input = el('input', {
        type: 'search', class: 'q2vt-search-input', placeholder: 'Search features…',
        'aria-label': 'Search features', autocomplete: 'off', spellcheck: 'false',
    });
    const clearButton = el('button', {
        type: 'button', class: 'q2vt-search-clear', title: 'Clear search', 'aria-label': 'Clear search',
    }, icon('close'));
    const list = el('div', { class: 'q2vt-search-results', role: 'listbox', hidden: true });
    const box = el('div', { class: 'q2vt-search' }, [
        el('div', { class: 'q2vt-search-bar' }, [icon('search'), input, clearButton]),
        list,
    ]);

    function message(text) {
        list.replaceChildren(el('div', { class: 'q2vt-search-message', text }));
        list.hidden = false;
    }

    function ensureIndex() {
        if (!loading) {
            loading = fetch(url)
                .then((r) => {
                    if (!r.ok) throw new Error(String(r.status));
                    return r.json();
                })
                .then((data) => {
                    index = new SearchIndex(data);
                    return index;
                })
                .catch(() => null);
        }
        return loading;
    }

    function choose(i) {
        if (!index || !results[i]) return;
        const item = index.describe(results[i]);
        input.value = item.value;
        list.hidden = true;
        onSelect(item.bbox, item.zooms);
    }

    function render() {
        const query = input.value;
        clearButton.hidden = !query;
        if (!query.trim()) {
            list.hidden = true;
            return;
        }
        if (!index) {
            message(loading ? 'Loading the search index…' : 'Search is not available for this map.');
            return;
        }
        results = index.find(query);
        active = results.length ? 0 : -1;
        if (!results.length) {
            message('No matching features.');
            return;
        }
        const shown = results.slice(0, MAX_RESULTS).map((r, i) => {
            const item = index.describe(r);
            return el('div', {
                class: `q2vt-search-item${i === active ? ' active' : ''}`, role: 'option',
                onmousedown: (e) => { e.preventDefault(); choose(i); },
            }, [
                el('div', { class: 'q2vt-search-value', html: highlight(item.value, query) }),
                el('div', { class: 'q2vt-search-where', text: `${item.layer} · ${item.field}` }),
            ]);
        });
        const extra = results.length > MAX_RESULTS
            ? el('div', { class: 'q2vt-search-message', text: `Showing ${MAX_RESULTS} of ${results.length} matches.` })
            : null;
        const note = index.truncated
            ? el('div', { class: 'q2vt-search-message', text: 'The index holds only part of this map’s features.' })
            : null;
        list.replaceChildren(...[...shown, extra, note].filter(Boolean));
        list.hidden = false;
    }

    function move(step) {
        const items = list.querySelectorAll('.q2vt-search-item');
        if (!items.length) return;
        items[active]?.classList.remove('active');
        active = (active + step + items.length) % items.length;
        items[active].classList.add('active');
        items[active].scrollIntoView({ block: 'nearest' });
    }

    let timer = null;
    input.addEventListener('focus', () => {
        ensureIndex().then(() => { if (input.value.trim()) render(); });
        if (input.value.trim()) render();
    });
    input.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            ensureIndex().then(render);
            render();
        }, 120);
    });
    input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
        else if (e.key === 'Enter') { e.preventDefault(); choose(Math.max(active, 0)); }
        else if (e.key === 'Escape') { list.hidden = true; input.blur(); }
    });
    input.addEventListener('blur', () => { list.hidden = true; });
    clearButton.hidden = true;
    clearButton.addEventListener('click', () => {
        input.value = '';
        results = [];
        list.hidden = true;
        clearButton.hidden = true;
        onClear();
    });

    return box;
}
