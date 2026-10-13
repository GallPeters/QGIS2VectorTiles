// Builds the web viewer into resources/viewer/: one viewer.html (from
// src/viewer.html) that loads either map library, switchable in the page:
//   ml/  bundle.js + bundle.css (shared controls + MapLibre adapter),
//        maplibre-gl.mjs, maplibre-gl-worker.mjs, maplibre-gl.css,
//        mapbox-gl-rtl-text.js (right-to-left labels)
//   ol/  bundle.js + bundle.css (shared controls + OpenLayers), ol.css
//
//   npm install      (once: installs the versions pinned in package.json)
//   npm run build
//
// To upgrade, e.g.:  npm install --save-exact ol@latest ol-mapbox-style@latest
// then build.

import { copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const here = dirname(fileURLToPath(import.meta.url));
const resources = join(here, '..', '..', 'resources', 'viewer');
const modules = join(here, 'node_modules');

// MapLibre stays a separate file next to the bundle: it finds its worker
// from its own URL, which bundling would break.
const maplibreExternal = {
    name: 'maplibre-external',
    setup(b) {
        b.onResolve({ filter: /^\.\/maplibre-gl\.mjs$/ }, (args) => ({ path: args.path, external: true }));
    },
};

const common = { bundle: true, minify: true, format: 'esm', target: 'es2022', logLevel: 'info' };

await build({
    ...common,
    entryPoints: { bundle: join(here, 'src', 'maplibre', 'main.js') },
    outdir: join(resources, 'ml'),
    plugins: [maplibreExternal],
});
await build({
    ...common,
    entryPoints: { bundle: join(here, 'src', 'openlayers', 'main.js') },
    outdir: join(resources, 'ol'),
});

const copies = [
    ['maplibre-gl/dist/maplibre-gl.mjs', 'ml/maplibre-gl.mjs'],
    ['maplibre-gl/dist/maplibre-gl-worker.mjs', 'ml/maplibre-gl-worker.mjs'],
    ['maplibre-gl/dist/maplibre-gl.css', 'ml/maplibre-gl.css'],
    ['@mapbox/mapbox-gl-rtl-text/dist/mapbox-gl-rtl-text.js', 'ml/mapbox-gl-rtl-text.js'],
    ['@mapbox/mapbox-gl-rtl-text/LICENSE.md', 'ml/mapbox-gl-rtl-text.LICENSE.md'],
    ['ol/ol.css', 'ol/ol.css'],
];
for (const [from, to] of copies) {
    copyFileSync(join(modules, from), join(resources, to));
    console.log(`Copied ${to}`);
}
copyFileSync(join(here, 'src', 'viewer.html'), join(resources, 'viewer.html'));
console.log('Copied viewer.html');
