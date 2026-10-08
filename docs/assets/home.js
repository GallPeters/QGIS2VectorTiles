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

  // --- Dialogs: close button and backdrop click ---------------------------
  document.querySelectorAll('dialog').forEach(function (dlg) {
    dlg.addEventListener('click', function (e) {
      if (e.target === dlg || e.target.closest('.dialog-close')) dlg.close();
    });
  });
})();
