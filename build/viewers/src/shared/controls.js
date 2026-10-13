// Small form controls for the settings panel: an on/off switch and a
// segmented control (one of a few options, all visible).

import { el } from './dom.js';

/** An on/off switch; ``onChange(on)`` is called when it is flipped. */
export function createToggle({ label, on = false, onChange }) {
    const button = el('button', {
        type: 'button', class: 'q2vt-toggle', role: 'switch',
        'aria-checked': String(on), 'aria-label': label,
        onclick: () => set(!on, true),
    }, el('span', { class: 'q2vt-toggle-knob' }));
    function set(value, notify) {
        on = value;
        button.setAttribute('aria-checked', String(on));
        if (notify) onChange(on);
    }
    return { element: button, set: (value) => set(value, false) };
}

/**
 * A segmented control. ``options`` are {id, label, title?}; ``onChoose(id)``
 * is called when another option is picked.
 */
export function createSegmented({ label, options, current, onChoose }) {
    const group = el('div', { class: 'q2vt-segmented', role: 'radiogroup', 'aria-label': label });
    function render() {
        group.replaceChildren(...options.map((option) => el('button', {
            type: 'button', role: 'radio',
            class: `q2vt-segment${option.id === current ? ' active' : ''}`,
            'aria-checked': String(option.id === current),
            title: option.title || option.label,
            onclick: () => {
                if (option.id === current) return;
                current = option.id;
                render();
                onChoose(option.id);
            },
        }, option.label)));
    }
    render();
    return { element: group };
}

/** A settings row: a label on the left, its control on the right. */
export function settingsRow(label, control) {
    return el('div', { class: 'q2vt-setting' }, [el('span', { class: 'q2vt-setting-label' }, label), control]);
}
