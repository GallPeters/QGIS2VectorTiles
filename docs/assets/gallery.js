/* QGIS2VectorTiles website: the home page's gallery of example web maps.
   The cards come from gallery.yaml; each example's web map sits in
   docs/examples/<id>/. */
(function () {
  'use strict';

  var gallery = document.getElementById('gallery');
  if (!gallery) return;
  var firstScreen = gallery.querySelector('.screen');
  var canHover = window.matchMedia('(hover: hover)').matches;

  var el = function (tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };
  var icon = function (name) {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    var use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    svg.setAttribute('aria-hidden', 'true');
    use.setAttribute('href', '#i-' + name);
    svg.appendChild(use);
    return svg;
  };
  var notice = function (text) {
    firstScreen.appendChild(el('p', 'gallery-note', text));
  };
  // 827 → "827 MB", 3.4 → "3.4 MB", 1530 → "1.5 GB".
  var size = function (mb) {
    if (mb >= 1000) return (Math.round(mb / 100) / 10) + ' GB';
    return (mb >= 10 ? Math.round(mb) : Math.round(mb * 10) / 10) + ' MB';
  };
  var coords = function (bounds) {
    return 'W ' + bounds[0] + '°, S ' + bounds[1] + '°, E ' + bounds[2] + '°, N ' + bounds[3] + '°';
  };

  // --- Live maps ---------------------------------------------------------
  // A card shows its still until "Explore live" is clicked; then the web map
  // loads over it. Only one card is live at a time: opening another puts
  // the first back to its still. As in the hero, the map takes the pointer
  // only after the click, and lets go when the pointer leaves it, the page
  // scrolls, or a click lands outside, so wheel and swipe keep moving the page.
  var live = null;        // {map, frame}
  var liveSince = 0;
  var setActive = function (on) {
    if (!live || live.map.classList.contains('is-active') === on) return;
    live.map.classList.toggle('is-active', on);
    if (on) {
      liveSince = Date.now();
      live.frame.focus({ preventScroll: true });
    }
  };
  var unload = function () {
    if (!live) return;
    live.frame.remove();
    live.map.querySelector('.live-loading').remove();
    live.map.classList.remove('is-loaded', 'is-active', 'is-live');
    live = null;
  };
  var goLive = function (map, title) {
    if (live && live.map === map) { setActive(true); return; }
    unload();
    var frame = el('iframe');
    frame.src = map.dataset.src;
    frame.title = title + ' live web map';
    frame.setAttribute('allow', 'fullscreen');
    // Shown a moment after its page loads, once the map has drawn.
    frame.addEventListener('load', function () {
      setTimeout(function () { if (live && live.frame === frame) map.classList.add('is-loaded'); }, 1500);
    });
    var activate = map.querySelector('.live-activate');
    map.insertBefore(el('p', 'live-loading', 'Loading the live map…'), activate);
    map.insertBefore(frame, activate);
    map.classList.add('is-live');
    live = { map: map, frame: frame };
    setActive(true);
  };
  document.addEventListener('pointerdown', function (e) {
    if (live && !live.map.contains(e.target)) setActive(false);
  });
  // Scrolling right after the click (bringing the map into view) keeps it.
  var release = function () { if (Date.now() - liveSince > 600) setActive(false); };
  window.addEventListener('scroll', release, { passive: true });
  document.body.addEventListener('scroll', release, { passive: true });

  // --- Cards -------------------------------------------------------------
  var button = function (label, iconName, cls, href, title) {
    var b;
    if (href) {
      b = el('a', 'btn btn-sm ' + cls);
      b.href = href;
      if (/^https?:/.test(href)) { b.target = '_blank'; b.rel = 'noopener'; }
    } else {
      b = el('button', 'btn btn-sm ' + cls);
      b.type = 'button';
      b.disabled = true;
      if (title) b.title = title;
    }
    b.append(icon(iconName), label);
    return b;
  };

  // The web map's address parameters: MapLibre, and the example's view and
  // background, whatever the visitor used last on this site.
  var viewParams = function (ex) {
    var view = ex.view || {};
    var params = 'viewer=ml';
    if (view.basemap) params += '&basemap=' + view.basemap;
    if (view.zoom !== undefined) params += '&z=' + view.zoom;
    if (view.center) params += '&c=' + view.center.join(',');
    return params;
  };

  var card = function (ex) {
    var soon = !!ex.coming_soon;
    var extent = ex.extent || {};
    var article = el('article', 'webmap' + (soon ? ' is-soon' : ''));
    article.id = 'gallery-' + ex.id;

    // The preview, in a window frame like the hero's.
    var win = el('figure', 'window webmap-window');
    var bar = el('div', 'window-bar');
    bar.append(el('i'), el('i'), el('i'), el('span', null, soon ? 'New example' : extent.name ? ex.title + ' · ' + extent.name : ex.title));
    [].forEach.call(bar.querySelectorAll('i'), function (i) { i.setAttribute('aria-hidden', 'true'); });
    var map = el('div', 'live-map webmap-map');
    if (soon) {
      var placeholder = el('div', 'webmap-placeholder');
      placeholder.appendChild(icon('globe'));
      map.appendChild(placeholder);
    } else {
      var img = el('img');
      img.src = ex.image;
      img.alt = 'The ' + ex.title + ' web map';
      img.loading = 'lazy';
      img.width = 1200;
      img.height = 750;
      map.appendChild(img);
      map.dataset.src = ex.webmap + '?embed=1&' + viewParams(ex);
      var activate = el('button', 'live-activate');
      activate.type = 'button';
      activate.append(icon('pointer'), 'Explore live');
      activate.addEventListener('click', function () { goLive(map, ex.title); });
      map.appendChild(activate);
      if (canHover) map.addEventListener('mouseleave', function () { if (live && live.map === map) setActive(false); });
    }
    win.append(bar, map);

    var body = el('div', 'webmap-body');
    var head = el('h3', 'h3', ex.title);

    var stats = el('ul', 'webmap-stats');
    var stat = function (iconName, text, hint) {
      var li = el('li');
      li.append(icon(iconName), el('span', null, text));
      if (hint) li.title = hint;
      stats.appendChild(li);
      return li;
    };
    // One row: where, which zooms, how many layers, and how much smaller
    // the web map is than the project's data (the sizes on hover).
    if (extent.name) stat('globe', extent.name, extent.bounds ? coords(extent.bounds) : null);
    if (ex.zooms) stat('search', 'Zoom ' + ex.zooms[0] + '–' + ex.zooms[1], 'Zoom levels');
    if (ex.layers) stat('layers', ex.layers + (ex.layers === 1 ? ' layer' : ' layers'), 'Layers in the QGIS project');
    if (ex.size && ex.size.original && ex.size.output) {
      var saved = Math.round((1 - ex.size.output / ex.size.original) * 100);
      stat('feather', '−' + saved + '%', 'Web map ' + size(ex.size.output) + ', from ' + size(ex.size.original) + ' of project data')
        .classList.add('webmap-saving');
    }

    var desc = el('p', 'webmap-desc', ex.description || '');
    var tags = el('ul', 'webmap-tags');
    (ex.tags || []).forEach(function (t) { tags.appendChild(el('li', null, t)); });

    var project = ex.project || {};
    var actions = el('div', 'webmap-actions');
    actions.append(
      button('View full web map', 'arrow', 'btn-primary', soon ? null : ex.webmap + '?' + viewParams(ex), 'Coming soon'),
      button('Download project' + (project.url && project.size ? ' (' + size(project.size) + ')' : ''), 'download', 'btn-ghost',
        soon ? null : project.url, soon ? 'Coming soon' : 'The project download is coming soon')
    );

    body.append(head);
    if (stats.children.length) body.appendChild(stats);
    body.appendChild(desc);
    if (tags.children.length) body.appendChild(tags);
    body.appendChild(actions);
    article.append(win, body);
    return article;
  };

  // On phones each card is a screen of its own (see site.css); on wider
  // screens the screens are transparent and the cards share one grid.
  var render = function (examples) {
    examples.forEach(function (ex, i) {
      if (i === 0) { firstScreen.appendChild(card(ex)); return; }
      var screen = el('div', 'screen');
      var head = el('div', 'section-head screen-head');
      head.append(el('p', 'eyebrow', 'Gallery'), el('p', 'h2', 'See it in action.'));
      screen.append(head, card(ex));
      gallery.appendChild(screen);
    });
  };

  if (window.location.protocol === 'file:') {
    notice('Browsers block loading gallery.yaml from a local file. Serve this folder instead, e.g. run "python -m http.server" here and open http://localhost:8000/.');
    return;
  }
  fetch('gallery.yaml')
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    })
    .then(function (text) {
      var data = window.jsyaml.load(text) || {};
      render(data.examples || []);
    })
    .catch(function (error) {
      console.warn('Gallery not loaded:', error);
      notice('The gallery could not be loaded. Open the examples directly in docs/examples.');
    });
})();
