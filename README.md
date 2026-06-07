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

The output package includes the following components:

| Component    | Format               | Description                                                     |
| ------------ | -------------------- | --------------------------------------------------------------- |
| **Tiles**    | `mbtiles`            | Vector tile dataset generated from QGIS layers and data sources |
| **Style**    | `json`               | MapLibre-compatible style specification                         |
| **Sprite**   | `png` + `json`       | Symbol atlas and metadata for map icons                         |
| **Viewer**   | `html`               | Ready-to-use MapLibre/OpenLayers web viewer                     |
| **Server**   | `py`                 | Local tile and style server                                     |
| **Launcher** | `bat` + `vbs` / `sh` | Platform-specific scripts to start server and open viewer       |

*Optional: generated only when required by the style.*


### Motivation

The plugin enables publishing lightweight client-side web maps directly from QGIS.

This workflow bridges the gap between desktop GIS, tile generation, and web mapping frameworks:

| Component                  | Strength                                  | Gap                         |
| -------------------------- | ----------------------------------------- | --------------------------- |
| **QGIS**                   | Advanced cartography and GIS design tools | No native web publishing    |
| **GDAL**                   | High-performance vector tile generation   | Command-line workflow       |
| **Web mapping frameworks** | Fast client-side rendering                | Limited cartographic design |
| **QGIS2VectorTiles**       | End-to-end desktop-to-web workflow        | —                           |


### Use Cases

| Use Case                     | Description                                                           | Technologies                                       |
| ---------------------------- | --------------------------------------------------------------------- | -------------------------------------------------- |
| **Web mapping applications** | Client-side vector tile rendering                                     | MapLibre, OpenLayers, Leaflet, MapTiler, Mapbox    |
| **Map services**             | Publishing via standard OGC web services                              | GeoServer, QGIS Server (MBTiles, Mapbox GL styles) |
| **Project distribution**     | Packaging complex cartographic outputs into a single portable dataset | —                                                  |
