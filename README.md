<div align="center">

<img width="100" alt="icon" src="https://github.com/user-attachments/assets/3dde55e2-9441-4ad6-b29f-d9edb786742e" />

  # QGIS2VectorTiles

 **From styled QGIS projects to client-side web maps in just one click.**

[![Issues](https://img.shields.io/badge/Issues-🛠️-f59e0b?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-🚀-7ec44a?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-📜-3b82f6?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)
  
</div>

### Introduction

QGIS2VectorTiles generates a vector tile package directly from the current QGIS project.

The plugin is designed to bridge the gap between the rich and highly flexible environment of **QGIS Desktop** —supporting complex spatial expressions, a wide range of data formats, geometry generators, and data-driven styling—and modern web mapping frameworks, which rely on client-side rendering of large datasets. The primary target ecosystem is **MapLibre** and compatible web mapping libraries.

To ensure portability and ease of installation, the tile generation process is based on the built-in **GDAL MBTiles driver**, avoiding external dependencies such as Tippecanoe. While Tippecanoe may offer higher performance in some scenarios, it requires separate installation and is primarily limited to Linux environments.

### Output Package

| Component     | Format         | Description                                                                 |
|---------------|----------------|-----------------------------------------------------------------------------|
| Tiles         | `mbtiles`      | Vector tile dataset generated from project layers and data sources.        |
| Style         | `json`         | Client-side style sheet following the MapLibre style specification and preserving QGIS styling as closely as possible. |
| Sprite*       | `png` + `json` | Icon package containing symbol images and metadata used by the client to resolve icons. |
| Viewer        | `html`         | Ready-to-use MapLibre/OpenLayers viewer referencing the tiles and style.   |
| Server        | `py`           | Local server that serves tiles and style resources locally.                |
| Launcher      | `bat` + `vbs` / `sh` | Platform-specific scripts for starting the local tile server and opening the viewer. |

\* Optional. Generated only when required by the style.


### Use Cases

* **Web mapping applications:** client-side rendering using MapLibre, OpenLayers, Mapbox and other web-mapping clients.
* **Map services:** WMS/WMTS publishing via QGIS Server or GeoServer.
* **Project distribution:** distribution of complex cartographic outputs from a single QGIS project and data source.

### Demo

https://github.com/user-attachments/assets/5e7c4518-ebe0-45fa-8659-e53fd67692fc
