// Bundles resources/ol_viewer/viewer.js with OpenLayers and ol-mapbox-style
// into resources/ol_viewer/bundle.js, and copies OpenLayers' ol.css next to it.
//
//   npm install      (once: installs the versions pinned in package.json)
//   npm run build
//
// To upgrade: npm install --save-exact ol@latest ol-mapbox-style@latest, then build.

import { copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const here = dirname(fileURLToPath(import.meta.url));
const viewer = join(here, '..', '..', 'resources', 'ol_viewer');

await build({
    entryPoints: [join(viewer, 'viewer.js')],
    outfile: join(viewer, 'bundle.js'),
    bundle: true,
    minify: true,
    format: 'esm',
    // viewer.js lives outside this folder; resolve its imports from here.
    nodePaths: [join(here, 'node_modules')],
    // The plugin fills _Q2VT_MINZOOM and _Q2VT_CENTER in when it serves the
    // map; they must reach the bundle as written.
    logLevel: 'info',
});
copyFileSync(join(here, 'node_modules', 'ol', 'ol.css'), join(viewer, 'ol.css'));
console.log('Copied ol.css');
