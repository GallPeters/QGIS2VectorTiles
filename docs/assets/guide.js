/* QGIS2VectorTiles website: styling guide, rendered from compatibility.yaml. */
(function () {
  'use strict';

  var content = document.getElementById('guide-content');
  var search = document.getElementById('guide-search');
  var count = document.getElementById('guide-count');
  var filters = document.querySelectorAll('[data-filter]');
  var level = 'all';

  var STATUS = {
    yes: 'Supported',
    partial: 'Partial',
    no: 'Not supported'
  };

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function norm(value) {
    return String(value || '').trim().toLowerCase();
  }

  function notice(message, isError) {
    content.replaceChildren(el('p', 'notice' + (isError ? ' error' : ''), message));
    content.setAttribute('aria-busy', 'false');
  }

  function cell(label, child) {
    var td = el('td');
    td.setAttribute('data-label', label);
    if (typeof child === 'string') td.textContent = child;
    else if (child) td.appendChild(child);
    return td;
  }

  function render(categories) {
    var frag = document.createDocumentFragment();
    categories.forEach(function (cat) {
      var section = el('section', 'category');
      var head = el('div', 'category-head');
      var title = el('h2', null, cat.name);
      var props = cat.properties || [];
      head.append(title, el('span', 'category-count', props.length + (props.length === 1 ? ' property' : ' properties')));

      var table = el('table');
      var thead = el('thead');
      var hr = el('tr');
      ['QGIS property', 'MapLibre property', 'Support', 'Data-driven', 'Notes'].forEach(function (h) {
        var th = el('th', null, h);
        th.scope = 'col';
        hr.appendChild(th);
      });
      thead.appendChild(hr);

      var tbody = el('tbody');
      props.forEach(function (p) {
        var status = norm(p.supported);
        if (!STATUS[status]) status = 'partial';
        var target = String(p.maplibre_prop || '').trim();
        var tr = el('tr');
        tr.dataset.status = status;
        tr.dataset.search = norm([p.qgis_prop, p.maplibre_prop, p.notes].join(' '));
        var prop = cell('QGIS property', String(p.qgis_prop || ''));
        prop.className = 'prop';
        var dd = norm(p.data_driven) === 'yes';
        tr.append(
          prop,
          cell('MapLibre', el('span', 'code' + (target.toUpperCase() === 'N/A' ? ' na' : ''), target || 'N/A')),
          cell('Support', el('span', 'status ' + status, STATUS[status])),
          cell('Data-driven', el('span', 'dd' + (dd ? ' yes' : ''), dd ? 'Yes' : 'No')),
          cell('Notes', String(p.notes || ''))
        );
        tbody.appendChild(tr);
      });

      table.append(thead, tbody);
      var wrap = el('div', 'table-wrap');
      wrap.appendChild(table);
      section.append(head, wrap);
      frag.appendChild(section);
    });

    var empty = el('p', 'empty', 'No properties match your filter.');
    empty.hidden = true;
    empty.id = 'guide-empty';
    frag.appendChild(empty);

    content.replaceChildren(frag);
    content.setAttribute('aria-busy', 'false');
    apply();
  }

  function apply() {
    var query = norm(search.value);
    var shown = 0;
    content.querySelectorAll('.category').forEach(function (section) {
      var visible = 0;
      section.querySelectorAll('tbody tr').forEach(function (tr) {
        var ok = (level === 'all' || tr.dataset.status === level) &&
                 (!query || tr.dataset.search.indexOf(query) !== -1);
        tr.hidden = !ok;
        if (ok) visible += 1;
      });
      section.hidden = visible === 0;
      shown += visible;
    });
    var empty = document.getElementById('guide-empty');
    if (empty) empty.hidden = shown !== 0;
    count.textContent = shown + (shown === 1 ? ' property' : ' properties');
  }

  search.addEventListener('input', apply);
  filters.forEach(function (btn) {
    btn.addEventListener('click', function () {
      level = btn.dataset.filter;
      filters.forEach(function (b) { b.setAttribute('aria-pressed', String(b === btn)); });
      apply();
    });
  });

  if (window.location.protocol === 'file:') {
    notice('Browsers block loading compatibility.yaml from a local file. Serve this folder instead, e.g. run "python -m http.server" here and open http://localhost:8000/compatibility.html.', true);
    return;
  }

  fetch('compatibility.yaml')
    .then(function (r) {
      if (!r.ok) throw new Error('compatibility.yaml could not be loaded (HTTP ' + r.status + ').');
      return r.text();
    })
    .then(function (text) {
      if (!window.jsyaml) throw new Error('The YAML parser did not load.');
      var data = window.jsyaml.load(text) || {};
      if (!Array.isArray(data.categories)) throw new Error('compatibility.yaml has no "categories" list.');
      render(data.categories);
    })
    .catch(function (err) { notice(err.message, true); });
})();
