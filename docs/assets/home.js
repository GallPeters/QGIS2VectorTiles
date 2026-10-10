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
      bullets: ['Coordinates multi-threaded tile slicing.', 'Hooks directly into system-level GDAL installations.', 'Runs optimized map routines.']
    },
    styler: {
      file: 'core/tiles_styler.py',
      bullets: ['Transforms styling definitions.', 'Matches and binds QGIS styles to vector channels.', 'Outputs unified .qlr layers.']
    },
    converter: {
      file: 'core/maplibre_converter.py',
      bullets: ['Translates QGIS configurations to MapLibre JSON.', 'Maps coordinate grids, fonts, and limits.', 'Generates icon sprite sheets.']
    },
    server: {
      file: 'core/server_initializer.py',
      bullets: ['Bundles output formats and local resources.', 'Constructs isolated deployment environments.', 'Provides instant web preview frameworks.']
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

  // --- Dialogs: close button and backdrop click ---------------------------
  document.querySelectorAll('dialog').forEach(function (dlg) {
    dlg.addEventListener('click', function (e) {
      if (e.target === dlg || e.target.closest('.dialog-close')) dlg.close();
    });
  });
})();
