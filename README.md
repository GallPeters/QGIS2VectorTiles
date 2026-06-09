<div align="center">

<img width="70" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![🐞 Issues](https://img.shields.io/badge/Issues-🐞-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![🌐 Website](https://img.shields.io/badge/Website-🌐-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![📜 License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

**From QGIS projects to ready-to-use web maps in one click**

<kbd>
<img width="500" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" />
</kbd>

</div>

## 📜 Introduction

### 🌍 Overview

**QGIS2VectorTiles** is a QGIS plugin that converts styled QGIS projects into **fast, client-rendered web maps**.  
Project data is compressed into lightweight **vector tiles**, while QGIS styling is converted into a client-side style that closely matches the original project.

The generated data and style are packaged together into a **fully standalone web map** — *no server setup, internet connection, or additional libraries required*.

### ⚡ Quick Start

**1. Install** the plugin from the **[QGIS Plugin Repository](https://plugins.qgis.org/plugins/QGIS2VectorTiles/)** or via the built-in Plugin Manager.  
**2. Run** the processing tool.  
**3. Launch** it — within seconds, your browser will automatically open an **interactive, high-performance web version** of your QGIS project.

### 🔄 Workflow

- 🎨 **Design in QGIS** — Advanced desktop cartography and styling  
- ⚙️ **Process with GDAL** — Powerful and scalable vector tile generation  
- 🗺️ **Render with MapLibre** — Fast and sharp client-side web maps  

### ✨ Key Features

- 🧩 **Advanced cartography** — Supports QGIS expressions, geometry generators, multi-layer symbology, and more  
- 🗂️ **Wide format support** — Compatible with GeoPackage, PostGIS, FlatGeobuf, GeoParquet, and other QGIS-supported formats  
- 🚀 **Optimized output** — Tiles and styles contain only the data and rules required for rendering  
- 📦 **Standalone package** — No server deployment, internet connection, or third-party installation required  

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

- 🖥️ **Client-side maps** — build modern web maps using **[MapLibre](https://maplibre.org/)**, **[OpenLayers](https://openlayers.org/)**, **[MapTiler](https://www.maptiler.com/)**, or **[Mapbox](https://www.mapbox.com/)**  
- 🌐 **OGC map services** — publish standard web map services via **[GeoServer](https://geoserver.org/)** or **[QGIS Server](https://qgis.org/)**  
- 📦 **Project distribution** — package a QGIS project into a single styled dataset  

_* Using the **[MBTiles](https://docs.geoserver.org/main/en/user/community/mbtiles/)** and **[MBStyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/)** extensions._

## 🛠️ Support

- 🏷️ Found an issue or rendering mismatch? Please **[open an issue](https://github.com/GallPeters/QGIS2VectorTiles/issues)** — feedback and bug reports are very welcome.
- 🌐 Updates and releases are available on my **[Mastodon](https://mastodon.social/@JossefKanter)** account.
