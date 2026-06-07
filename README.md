<div align="center">

<img width="90" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![Issues](https://img.shields.io/badge/Issues-🛠️-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-🚀-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

**From styled QGIS projects to client-side web maps in a single workflow.**

<img width="500" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" />

</div>

### Introduction

QGIS2VectorTiles generates a styled vector tile package directly from a QGIS project.

It enables publishing lightweight client-side web maps without leaving the QGIS environment. The plugin unifies the workflow between QGIS, GDAL, and modern web mapping frameworks such as MapLibre and OpenLayers.

| Technology                | Strength                                  | Gap                           |
| ------------------------- | ----------------------------------------- | ----------------------------- |
| **QGIS**                  | Advanced cartography and GIS design tools | Web map publishing            |
| **GDAL**                  | High-performance vector tile generation   | Command-line workflow         |
| **MapLibre / OpenLayers** | Fast client-side map rendering            | Limited cartographic modeling |
| **QGIS2VectorTiles**      | End-to-end desktop-to-web workflow        | —                             |


### Output Package

| Component    | Format               | Description                                                     |
| ------------ | -------------------- | --------------------------------------------------------------- |
| **Tiles**    | `mbtiles`            | Vector tile dataset generated from QGIS layers and data sources |
| **Style**    | `json`               | MapLibre-compatible style specification                         |
| **Sprite**   | `png` + `json`       | Symbol atlas and metadata for map icons                         |
| **Viewer**   | `html`               | Ready-to-use MapLibre/OpenLayers web viewer                     |
| **Server**   | `py`                 | Local tile and style server                                     |
| **Launcher** | `bat` + `vbs` / `sh` | Platform-specific scripts to start server and open viewer       |

* Optional. Generated only when required by the style.

### Use Cases

* **Web mapping applications** – Client-side vector tile rendering using libraries such as [MapLibre](https://maplibre.org), [OpenLayers](https://openlayers.org), [Leaflet](https://leafletjs.com), [MapTiler](https://www.maptiler.com), and [Mapbox](https://www.mapbox.com).

* **Map services** – Publishing via standard map servers such as [GeoServer](https://geoserver.org) or [QGIS Server](https://qgis.org) using MBTiles and Mapbox GL styles.

* **Project distribution** – Packaging complex cartographic outputs into a single portable, styled dataset.
