# Web map screenshot check

Exports projects with this repository's plugin, takes MapLibre screenshots
of fixed views and compares them with reference images. It catches changes
in how the web map draws, for example after a MapLibre upgrade. The plugin
relies on some MapLibre behaviour that isn't documented: badges inside
labels, stretched backgrounds, and backgrounds with variable anchors.

## Setup (once)

1. Install Node.js, then in this folder: `npm install`.
2. Copy `views.example.json` to a file outside the repository, set your
   projects, extents and views, and set `references` to a folder outside the
   repository (the images show your data).

## Run

With QGIS's Python (Windows: OSGeo4W's `python-qgis.bat`):

    python-qgis.bat tools\visual_check\check.py C:\path\to\views.json --update   (record references)
    python-qgis.bat tools\visual_check\check.py C:\path\to\views.json            (compare)

Each view prints the share of pixels that differ. A view over its
tolerance has a `_diff.png` next to its screenshot (in `<references>/_run`),
with the differing pixels in red. `--only NAME` checks one project.

Run it before each release and before upgrading MapLibre, after recording
references with a version you have checked by eye.
