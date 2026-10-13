/* QGIS2VectorTiles website: home page interactions. */
(function () {
  'use strict';

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var canHover = window.matchMedia('(hover: hover)').matches;

  // --- QGIS / web comparison slider -------------------------------------
  // A transparent range input covers the images, so dragging, tapping and
  // the arrow keys all work natively and screen readers get a slider.
  var compare = document.querySelector('.compare');
  if (compare) {
    var range = compare.querySelector('.compare-range');
    var touched = false;
    var setPos = function (v) {
      compare.style.setProperty('--pos', v + '%');
      range.setAttribute('aria-valuetext', Math.round(100 - v) + '% web map shown');
    };
    range.addEventListener('input', function () { touched = true; setPos(range.value); });
    range.addEventListener('pointerdown', function () { touched = true; });
    setPos(range.value);

    // A short sweep the first time the slider is seen, so it reads as interactive.
    if (!reduceMotion && 'IntersectionObserver' in window) {
      var ease = function (t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
      var frames = [[50, 78, 900], [78, 24, 1100], [24, 50, 800]];
      var sweep = function () {
        var i = 0, start = null;
        var step = function (now) {
          if (touched) return;
          if (start === null) start = now;
          var f = frames[i], t = Math.min(1, (now - start) / f[2]);
          var v = f[0] + (f[1] - f[0]) * ease(t);
          range.value = v;
          setPos(v);
          if (t === 1) { i += 1; start = null; if (i === frames.length) return; }
          window.requestAnimationFrame(step);
        };
        setTimeout(function () { window.requestAnimationFrame(step); }, 600);
      };
      var seen = new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) { seen.disconnect(); sweep(); }
      }, { threshold: 0.5 });
      seen.observe(compare);
    }
  }

  // --- Pipeline module dialog --------------------------------------------
  var modules = {
    orchestrator: {
      file: 'qgis2vectortiles.py',
      bullets: ['Core plugin framework entry point script.', 'Manages operational properties and runtime variables.', 'Coordinates sequential execution across the workspace.']
    },
    flattener: {
      file: 'core/rules_flattener.py',
      bullets: ['Processes intricate desktop renderer rules.', 'Flattens compound parameters into clear conditional groups.', 'Preserves filters and scaling logic.']
    },
    exporter: {
      file: 'core/rules_exporter.py',
      bullets: ['Slices vector features into intermediate datastores.', 'Executes spatial bounds trimming.', 'Applies geometry correction schemas.']
    },
    generator: {
      file: 'core/tiles_generator.py',
      bullets: ['Slices the datasets into vector tiles with parallel GDAL runs, split by zoom level and area.', 'Uses the GDAL that ships with QGIS.', 'Writes all the tiles into one PMTiles file.']
    },
    styler: {
      file: 'core/tiles_styler.py',
      bullets: ['Transforms styling definitions.', 'Matches and binds QGIS styles to vector channels.', 'Hands one styled layer to the MapLibre converter.']
    },
    converter: {
      file: 'core/maplibre_converter.py',
      bullets: ['Translates QGIS configurations to MapLibre JSON.', 'Maps coordinate grids, fonts, and limits.', 'Generates icon sprite sheets.']
    },
    server: {
      file: 'core/server_initializer.py',
      bullets: ['Lays out the output folder as a complete static web map: index.html, style and tiles.pmtiles.', 'Starts a small local server at the end of each run and opens the viewer in your browser.', 'The same folder can be uploaded to GitHub Pages or any web host as it is.']
    }
  };
  var repo = 'https://github.com/GallPeters/QGIS2VectorTiles/blob/main/src/';
  var moduleDialog = document.getElementById('module-dialog');
  if (moduleDialog) {
    var title = moduleDialog.querySelector('h2');
    var path = moduleDialog.querySelector('.mono');
    var list = moduleDialog.querySelector('ul');
    var link = moduleDialog.querySelector('a');
    document.querySelectorAll('.step[data-module]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var m = modules[btn.dataset.module];
        if (!m) return;
        title.textContent = m.file.split('/').pop();
        path.textContent = 'src/' + m.file;
        list.replaceChildren.apply(list, m.bullets.map(function (b) {
          var li = document.createElement('li');
          li.textContent = b;
          return li;
        }));
        link.href = repo + m.file;
        moduleDialog.showModal();
      });
    });
  }

  // --- Demo gallery ------------------------------------------------------
  // Posters are set when the gallery approaches the viewport; a video loads
  // only when its card is hovered (preview) or opened (full player).
  var demos = document.querySelectorAll('.demo');
  if (demos.length) {
    var setPosters = function () {
      demos.forEach(function (d) {
        var v = d.querySelector('video');
        if (v && v.dataset.poster) { v.poster = v.dataset.poster; v.removeAttribute('data-poster'); }
      });
    };
    if ('IntersectionObserver' in window) {
      var near = new IntersectionObserver(function (entries) {
        if (entries.some(function (e) { return e.isIntersecting; })) { near.disconnect(); setPosters(); }
      }, { rootMargin: '600px 0px' });
      demos.forEach(function (d) { near.observe(d); });
    } else {
      setPosters();
    }

    if (canHover && !reduceMotion) {
      demos.forEach(function (d) {
        var v = d.querySelector('video');
        d.addEventListener('mouseenter', function () {
          if (v.dataset.src) { v.src = v.dataset.src; v.removeAttribute('data-src'); }
          v.play().catch(function () {});
        });
        d.addEventListener('mouseleave', function () { v.pause(); });
      });
    }

    var videoDialog = document.getElementById('video-dialog');
    var player = videoDialog && videoDialog.querySelector('video');
    demos.forEach(function (d) {
      d.addEventListener('click', function () {
        var v = d.querySelector('video');
        v.pause();
        player.src = d.dataset.video;
        player.setAttribute('aria-label', d.querySelector('.h3').textContent);
        videoDialog.showModal();
        player.play().catch(function () {});
      });
    });
    if (videoDialog) {
      videoDialog.addEventListener('close', function () {
        player.pause();
        player.removeAttribute('src');
        player.load();
      });
    }
  }

  // --- Section dots ------------------------------------------------------
  // One link per full-screen section; the browser's scroll snapping does the
  // moving, the dots show where you are and jump on click.
  var dotsNav = document.querySelector('.dots');
  var slides = document.querySelectorAll('.slide[data-title]');
  if (dotsNav && slides.length) {
    var dotFor = {};
    slides.forEach(function (slide) {
      var a = document.createElement('a');
      a.href = '#' + slide.id;
      a.dataset.label = slide.dataset.title;
      a.setAttribute('aria-label', slide.dataset.title);
      dotsNav.appendChild(a);
      dotFor[slide.id] = a;
    });
    var activeDot = null;
    var setActive = function (id) {
      var dot = dotFor[id];
      if (!dot || dot === activeDot) return;
      if (activeDot) { activeDot.classList.remove('is-active'); activeDot.removeAttribute('aria-current'); }
      dot.classList.add('is-active');
      dot.setAttribute('aria-current', 'true');
      activeDot = dot;
    };
    setActive(slides[0].id);
    if ('IntersectionObserver' in window) {
      var spy = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) { if (entry.isIntersecting) setActive(entry.target.id); });
      }, { rootMargin: '-45% 0px -45% 0px' });
      slides.forEach(function (slide) { spy.observe(slide); });
    }
  }

  // --- Desktop: one wheel step glides to the next section ----------------
  // With mandatory snapping, a wheel step first scrolls a little and the
  // browser snaps the rest a moment later, which feels jerky. Here a wheel
  // step scrolls straight to the next section in one smooth move, as Page
  // Down does. A section taller than the window scrolls normally until its
  // edge; a trackpad swipe and its fading momentum move one section only.
  var snapScreens = window.matchMedia('(min-width: 1025px) and (min-height: 540px)');
  var allSlides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  if (allSlides.length > 1) {
    var gliding = false;
    var glideTimer = 0;
    var lastWheel = 0;
    var lastDelta = 0;
    var endGlide = function () { gliding = false; clearTimeout(glideTimer); };
    window.addEventListener('scrollend', endGlide);
    var pageTop = function (el) { return el.getBoundingClientRect().top + window.scrollY; };
    var inScrollable = function (el) {
      for (; el && el !== document.body; el = el.parentElement) {
        if (el.scrollHeight > el.clientHeight + 1 && /(auto|scroll)/.test(getComputedStyle(el).overflowY)) return true;
      }
      return false;
    };
    window.addEventListener('wheel', function (e) {
      if (!snapScreens.matches || e.ctrlKey || e.defaultPrevented) return;
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX) || document.querySelector('dialog[open]')) return;
      if (inScrollable(e.target)) return;
      var now = Date.now();
      var delta = Math.abs(e.deltaY);
      // A new step: after a pause, or a full wheel-notch-sized one. Small,
      // quick events are a trackpad swipe still going on.
      var newStep = now - lastWheel > 150 || (delta >= 40 && delta >= lastDelta);
      lastWheel = now;
      lastDelta = delta;
      if (gliding) {
        e.preventDefault();
        return;
      }
      var dir = e.deltaY > 0 ? 1 : -1;
      var y = window.scrollY;
      var current = 0;
      allSlides.forEach(function (slide, i) { if (pageTop(slide) <= y + 4) current = i; });
      var slideTop = pageTop(allSlides[current]);
      var slideBottom = slideTop + allSlides[current].offsetHeight;
      if (dir > 0 && slideBottom > y + window.innerHeight + 4) return;
      if (dir < 0 && slideTop < y - 4) return;
      var target = allSlides[current + dir];
      if (!target) return;
      e.preventDefault();
      if (!newStep) return;
      gliding = true;
      // Browsers without the scrollend event end the glide by time.
      glideTimer = setTimeout(endGlide, 900);
      window.scrollTo({ top: pageTop(target), behavior: reduceMotion ? 'auto' : 'smooth' });
    }, { passive: false });
  }

  // --- Phones: feature and use-case cards open their full text -----------
  // On the one-screen phone layout the cards show only their title; a tap
  // opens a dialog with a copy of the whole card.
  var paged = window.matchMedia('(max-width: 1024px) and (min-height: 480px)');
  var infoDialog = document.getElementById('info-dialog');
  var infoCards = document.querySelectorAll('.feature, .usecase');
  if (infoDialog && infoCards.length) {
    var infoBody = infoDialog.querySelector('.dialog-inner');
    var syncCards = function () {
      infoCards.forEach(function (card) {
        if (paged.matches) {
          card.setAttribute('role', 'button');
          card.setAttribute('tabindex', '0');
          card.setAttribute('aria-haspopup', 'dialog');
        } else {
          card.removeAttribute('role');
          card.removeAttribute('tabindex');
          card.removeAttribute('aria-haspopup');
        }
      });
    };
    var openCard = function (card) {
      infoBody.replaceChildren.apply(infoBody, [].map.call(card.children, function (el) { return el.cloneNode(true); }));
      infoDialog.setAttribute('aria-label', card.querySelector('.h3').textContent);
      infoDialog.dataset.kind = card.classList.contains('feature') ? 'feature' : 'usecase';
      infoDialog.showModal();
    };
    syncCards();
    if (paged.addEventListener) paged.addEventListener('change', syncCards);
    infoCards.forEach(function (card) {
      card.addEventListener('click', function (e) {
        if (!paged.matches || e.target.closest('a')) return;
        openCard(card);
      });
      card.addEventListener('keydown', function (e) {
        if (!paged.matches || e.target !== card || (e.key !== 'Enter' && e.key !== ' ')) return;
        e.preventDefault();
        openCard(card);
      });
    });
  }

  // --- Hero: "Complex projects" panel -------------------------------------
  // A popover (top layer, so the hero never clips it) listing the complex
  // features, with a search over the styling guide's data
  // (compatibility.yaml, loaded on first open). Wide screens place it under
  // its button; phones show it as a sheet below the header (CSS).
  var complexToggle = document.getElementById('complex-toggle');
  var complexPanel = document.getElementById('complex-panel');
  if (complexToggle && complexPanel) {
    var native = typeof complexPanel.showPopover === 'function';
    var searchBox = document.getElementById('complex-search');
    var featureList = document.getElementById('complex-list');
    var results = document.getElementById('complex-results');
    var guideLink = document.getElementById('complex-guide');
    var wide = window.matchMedia('(min-width: 641px)');
    var index = null;
    var loading = null;
    var STATUS = { yes: 'Supported', partial: 'Partial', no: 'Not supported' };
    // Everyday words for settings QGIS names otherwise.
    var SYNONYMS = {
      halo: ['buffer'], outline: ['stroke'], border: ['stroke'], transparency: ['opacity'],
      rotation: ['angle', 'rotation'], icon: ['marker', 'svg'], dash: ['dash', 'pattern']
    };

    var isOpen = function () {
      return native ? complexPanel.matches(':popover-open') : complexPanel.classList.contains('is-open');
    };
    var place = function () {
      if (!wide.matches) { complexPanel.style.top = complexPanel.style.left = complexPanel.style.maxHeight = ''; return; }
      var r = complexToggle.getBoundingClientRect();
      var width = Math.min(460, window.innerWidth - 24);
      var left = Math.min(Math.max(12, r.left), window.innerWidth - width - 12);
      var below = window.innerHeight - r.bottom - 16;
      var above = r.top - 16;
      complexPanel.style.left = left + 'px';
      complexPanel.style.width = width + 'px';
      if (below >= 360 || below >= above) {
        complexPanel.style.top = (r.bottom + 8) + 'px';
        complexPanel.style.maxHeight = below + 'px';
      } else {
        var height = Math.min(above, 560);
        complexPanel.style.top = (r.top - 8 - height) + 'px';
        complexPanel.style.maxHeight = height + 'px';
      }
    };
    var loadScript = function (src) {
      return new Promise(function (resolve, reject) {
        if (window.jsyaml) return resolve();
        var s = document.createElement('script');
        s.src = src;
        s.onload = resolve;
        s.onerror = function () { reject(new Error('The YAML parser did not load.')); };
        document.head.appendChild(s);
      });
    };
    var loadIndex = function () {
      if (loading) return loading;
      loading = Promise.all([
        loadScript('assets/vendor/js-yaml.min.js'),
        fetch('compatibility.yaml').then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.text();
        })
      ]).then(function (parts) {
        var data = window.jsyaml.load(parts[1]) || {};
        index = [];
        (data.categories || []).forEach(function (cat) {
          (cat.properties || []).forEach(function (p) {
            var status = String(p.supported || '').trim().toLowerCase();
            index.push({
              prop: String(p.qgis_prop || ''),
              cat: cat.name,
              status: STATUS[status] ? status : 'partial',
              notes: String(p.notes || ''),
              title: String(p.qgis_prop || '').toLowerCase(),
              text: [p.qgis_prop, p.maplibre_prop, p.notes, cat.name].join(' ').toLowerCase()
            });
          });
        });
        return index;
      });
      loading.catch(function () { loading = null; });
      return loading;
    };
    var el = function (tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text) node.textContent = text;
      return node;
    };
    var search = function () {
      var query = searchBox.value.trim().toLowerCase();
      guideLink.href = 'compatibility.html' + (query ? '?q=' + encodeURIComponent(searchBox.value.trim()) : '');
      featureList.hidden = !!query;
      results.hidden = !query;
      if (!query) return;
      if (!index) {
        results.replaceChildren(el('p', 'complex-empty', 'Loading the styling guide…'));
        loadIndex().then(search, function () {
          results.replaceChildren(el('p', 'complex-empty', 'The styling guide could not be loaded here. Open it below.'));
        });
        return;
      }
      var words = query.split(/\s+/);
      var hits = index.filter(function (item) {
        return words.every(function (w) { return item.text.indexOf(w) !== -1; });
      });
      // Settings named like the search (or its QGIS name) first.
      var named = [words[0]].concat(SYNONYMS[words[0]] || []);
      var score = function (item) {
        return named.some(function (w) { return item.title.indexOf(w) !== -1; }) ? 1 : 0;
      };
      hits.sort(function (a, b) { return score(b) - score(a); });
      if (!hits.length) {
        results.replaceChildren(el('p', 'complex-empty', 'No setting matches "' + searchBox.value.trim() + '". Try another word, or open the full guide.'));
        return;
      }
      var list = el('ul', 'complex-hits');
      hits.slice(0, 8).forEach(function (item) {
        var li = el('li');
        var a = el('a');
        a.href = 'compatibility.html?q=' + encodeURIComponent(item.prop);
        var head = el('span', 'hit-head');
        head.append(el('span', 'status ' + item.status, STATUS[item.status]), el('b', null, item.prop));
        a.append(head, el('span', 'hit-cat', item.cat), el('span', 'hit-notes', item.notes));
        li.appendChild(a);
        list.appendChild(li);
      });
      var nodes = [list];
      if (hits.length > 8) nodes.push(el('p', 'complex-more', (hits.length - 8) + ' more in the full guide.'));
      results.replaceChildren.apply(results, nodes);
    };
    var onToggled = function () {
      var open = isOpen();
      complexToggle.setAttribute('aria-expanded', String(open));
      if (open) {
        place();
        loadIndex().catch(function () {});
        if (wide.matches) searchBox.focus({ preventScroll: true });
      }
    };
    if (native) {
      complexToggle.setAttribute('popovertarget', 'complex-panel');
      complexPanel.addEventListener('toggle', onToggled);
      complexPanel.addEventListener('beforetoggle', function (e) { if (e.newState === 'open') place(); });
    } else {
      // Without popovers: toggled here, closed by Escape or a click outside.
      complexPanel.removeAttribute('popover');
      complexToggle.addEventListener('click', function () {
        complexPanel.classList.toggle('is-open');
        onToggled();
      });
      document.addEventListener('click', function (e) {
        if (isOpen() && !complexPanel.contains(e.target) && !complexToggle.contains(e.target)) {
          complexPanel.classList.remove('is-open');
          onToggled();
        }
      });
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && isOpen()) { complexPanel.classList.remove('is-open'); onToggled(); complexToggle.focus(); }
      });
    }
    var close = function () {
      if (!isOpen()) return;
      if (native) complexPanel.hidePopover();
      else { complexPanel.classList.remove('is-open'); onToggled(); }
    };
    // The page moved under it: close, as menus do.
    window.addEventListener('scroll', close, { passive: true });
    document.body.addEventListener('scroll', close, { passive: true });
    window.addEventListener('resize', function () { if (isOpen()) place(); });
    searchBox.addEventListener('input', search);
    featureList.addEventListener('click', function (e) {
      var btn = e.target.closest('button[data-q]');
      if (!btn) return;
      searchBox.value = btn.dataset.q;
      search();
      searchBox.focus({ preventScroll: true });
    });
  }

  // --- Dialogs: close button and backdrop click ---------------------------
  document.querySelectorAll('dialog').forEach(function (dlg) {
    dlg.addEventListener('click', function (e) {
      if (e.target === dlg || e.target.closest('.dialog-close')) dlg.close();
    });
  });
})();
