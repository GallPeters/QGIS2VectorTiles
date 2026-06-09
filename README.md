<div align="center">

<img width="70" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![🐛 Issues](https://img.shields.io/badge/Issues-🐛-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![🌐 Website](https://img.shields.io/badge/Website-🚀-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![📜 License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

**From QGIS projects to ready-to-use web maps in one click**

<kbd>
<img width="500" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" />
</kbd>

</div>

## Introduction

QGIS2VectorTiles Pack a styled QGIS projects into lightweight, ready-to-use, client-rendered web maps in a single click.

### Overview

> 🚀 **Install** via the **[QGIS Plugin Repository](https://plugins.qgis.org/plugins/QGIS2VectorTiles/)** or the built-in Plugin Manager.

- **QGIS Plugin** – packages a QGIS project into a styled, ready-to-use, client-rendered web map.
- **Processing Tool** – lets users control extent, zoom range, fields, and background type.
- **Output Map** – opens instantly in the browser and requires no network connection or additional dependencies.

### Key Features

> ⚡ **Tight data–style coupling** – only required data and style rules are sent to the client.

- **Design in QGIS** – supports expressions, geometry generators, and multiple data formats.  
- **Process with GDAL** – generates vector tiles using the GDAL MBTiles driver.  
- **Render in MapLibre** – client-side rendering via a MapLibre-based web viewer.  

## Generated Package

| Component | Format | Description |
| :--- | :--- | :--- |
| **Tiles** | `mbtiles` | Single vector tileset. |
| **Style** | `json` | MapLibre style sheet matching QGIS styling. |
| **Sprites** | `png + json` | Sprite sheet for markers and labels. |
| **Viewer** | `html + js + css` | Offline MapLibre / OpenLayers web viewer. |
| **Server** | `py` | Lightweight local Python server for tiles and styles. |
| **Launcher** | `bat + vbs / sh` | Script to start the server and open the browser. |

_* Only when marker symbols are used in the project._

## Use Cases

* **Client Side Maps** — build client-side maps using **[MapLibre](https://maplibre.org/)**, **[OpenLayers](https://openlayers.org/)**, **[MapTiler](https://www.maptiler.com/)** or **[Mapbox](https://www.mapbox.com/)**.  
* **OGC Map Services** — publish standard web map services via **[GeoServer*](https://geoserver.org/)** or **[QGIS Server](https://qgis.org/)**.  
* **Project Distribution** — package projects into a single styled layer and a single data source.  

_* Using the **[MBTiles](https://docs.geoserver.org/main/en/user/community/mbtiles/)** and **[MBStyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/)** extensions._

## Support

🏷️ Found an issue or QGIS–MapLibre mismatch? Please **[open an issue](https://github.com/GallPeters/QGIS2VectorTiles/issues)** and I’ll do my best to help you resolve it.

🌐 Updates and releases are available on my **[Mastodon](https://mastodon.social/@JossefKanter)** account.
