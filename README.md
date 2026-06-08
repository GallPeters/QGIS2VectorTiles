<div align="center">

<img width="90" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![Issues](https://img.shields.io/badge/Issues-🛠️-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-🚀-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

<h3>From styled QGIS projects to ready-to-use web maps in a single click</h3>

<kbd>
  <img width="650" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" />
</kbd>

</div>



## Introduction

**QGIS2VectorTiles** bridges the gap between desktop cartography and modern web mapping. Design maps using native QGIS tools (including expression-based styling, geometry generators, and complex layouts) and export them into fully styled client-side packages with a single click.

> **Installation:** Available via the official **[QGIS Plugin Repository](https://plugins.qgis.org/)** or directly through the built-in QGIS **Plugin Manager**.

### Core Engine Advantages
* **Optimized Data** - Lightweight vector tiles generated using the high-performance GDAL CLI engine.
* **Seamless Styling** - A MapLibre-compatible style sheet that mirrors your original QGIS layer hierarchy.

The plugin enforces a **tight coupling between data and styling**, guaranteeing that client web applications only request the exact geometry attributes and corresponding rendering rules required for the view.



## Generated Package

| Component | Format | Description |
| :--- | :--- | :--- |
| **Tiles** | `mbtiles` | A single, high-performance vector tile dataset. |
| **Style** | `json` | MapLibre style sheet matching the original QGIS project design. |
| **Sprites** | `png` + `json` | Automatically compiled marker symbol package (generated only when required). |
| **Viewer** | `html` + `js` + `css` | Ready-to-use, zero-configuration offline MapLibre / OpenLayers web viewer. |
| **Server** | `py` | Ultra-lightweight local server to test and serve your tiles instantly. |
| **Launcher**| `bat` + `vbs` / `sh` | Platform-specific scripts to spin up the server and launch the browser automatically. |



## Key Use Cases

* **Web Mapping Applications** - Deploy high-performance client-side maps using **[MapLibre](https://maplibre.org/)**, **[OpenLayers](https://openlayers.org/)**, **[MapTiler](https://www.maptiler.com/)**, or **[Mapbox](https://www.mapbox.com/)**.
* **OGC Map Services** - Publish production-grade tile layers through **[GeoServer](https://geoserver.org/)** (using the **[MBTiles](https://docs.geoserver.org/main/en/user/community/mbtiles/)** and **[MBStyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/)** extensions) or natively via **[QGIS Server](https://qgis.org/)**.
* **Portable Project Distribution** - Package heavy, multi-source cartographic environments into a single, offline-ready file structure complete with an integrated local runtime.



## Support & Community

Encountered a rendering discrepancy or have a feature request? Please **[open an issue](https://github.com/GallPeters/QGIS2VectorTiles/issues)** - contributions and feedback are always welcome!

Follow project updates and new releases on my **[Mastodon](https://mastodon.social/@JossefKanter)** account.
