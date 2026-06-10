<div align="center">

<img width="70" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![🐞 Issues](https://img.shields.io/badge/Issues-🐞-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![🌐 Website](https://img.shields.io/badge/Website-🌐-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![📜 License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

**From QGIS projects to fast, lightweight, client-rendered web maps in a single run.**

<kbd>
<img width="630" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" />
</kbd>

</div>

## 📜 Introduction

### 🌍 Overview

QGIS2VectorTiles is a QGIS plugin that converts QGIS projects into client-rendered web maps.
- **Data** is compressed into a fast, lightweight **vector tiles**.
- **Style** is converted into a **client-side style** that closely matches the original project.
- **Package** contains the generated data and style as a **fully offline standalone web map**.   

### ⚡ Quick Start

1. **Install** the plugin from the **[QGIS Plugin Repository](https://plugins.qgis.org/plugins/QGIS2VectorTiles/)** or via the built-in Plugin Manager.
2. **Run** the processing tool and wait a few seconds after it finishes.
3. **View** a web version of your QGIS project that opens automatically in your browser.

### 🔄 Workflow

- **Design in QGIS** - Advanced desktop cartography and styling.  
- **Process with GDAL** - Powerful and scalable vector tile generation.  
- **Render with MapLibre** - Fast and sharp client-side web maps.  

### ✨ Key Features

- **Advanced cartography** - Supports QGIS expressions, geometry generators and more.  
- **Wide format support** - Compatible with QGIS-supported formats as GeoPackage and Parquets.  
- **Optimized output** - Tiles and styles contain only the data and rules required for rendering.  
- **Standalone package** - No server deployment, internet connection, or third-party installation required.  

## 📦 Generated Package

| Component | Format | Description |
| :--- | :--- | :--- |
| **Tiles** | `mbtiles` | Single vector tileset |
| **Style** | `json` | MapLibre style sheet matching QGIS styling |
| **Sprites** | `png + json` | Sprite sheet for markers and labels |
| **Viewer** | `html + js + css` | Offline MapLibre / OpenLayers web viewer |
| **Server** | `py` | Lightweight local Python server for tiles and styles |
| **Launcher** | `bat + vbs / sh` | Script to start the server and open the browser |

_* Only when marker symbols are used in the project._

## 🚀 Use Cases

- **Client-side maps** - build modern web maps using **[MapLibre](https://maplibre.org/)**, **[OpenLayers](https://openlayers.org/)**, **[MapTiler](https://www.maptiler.com/)**, or **[Mapbox](https://www.mapbox.com/)**  
- **OGC map services** - publish standard web map services via **[GeoServer](https://geoserver.org/)** or **[QGIS Server](https://qgis.org/)**  
- **Project distribution** - package a QGIS project into a single styled dataset  

_* Using the **[MBTiles](https://docs.geoserver.org/main/en/user/community/mbtiles/)** and **[MBStyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/)** extensions._

## 🛠️ Support

- **Found a mismatch?** Please **[open an issue](https://github.com/GallPeters/QGIS2VectorTiles/issues)** - feedback and bug reports are very welcome.
- **Want to get updates?** Releases are available on my **[Mastodon](https://mastodon.social/@JossefKanter)** account.
