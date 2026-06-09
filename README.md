<div align="center">

<img width="70" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![Issues](https://img.shields.io/badge/Issues-🛠️-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-🚀-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

**From styled QGIS projects to ready-to-use web maps in one click**

<kbd><img width="500" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" /></kbd>

</div>

## Introduction

### Overview

> 🚀 **Installation:** Available via the **[QGIS Plugin Repository](https://plugins.qgis.org/plugins/QGIS2VectorTiles/)** or the QGIS **Plugin Manager**.

Pack a styled QGIS projects into lightweight, ready-to-use, client-rendered web maps in a single click.

### Key Features

> ⚡ **Tight data–style coupling** — only the required data and style rules are being sent to the client.

* **Design in QGIS** — support for expressions, geometry generators and a wide variety of data formats.  
* **Process with GDAL** — vector tiles generated using the GDAL MBTiles driver.  
* **View in MapLibre** — client-side rendering with a MapLibre-based web viewer.  

## Generated Package

| Component | Format | Description |
| :--- | :--- | :--- |
| **Tiles** | `mbtiles` | Single vector tile dataset. |
| **Style** | `json` | MapLibre style sheet matching the original QGIS project design. |
| **Sprites*** | `png` + `json` | Marker symbol package. |
| **Viewer** | `html` + `js` + `css` | Offline MapLibre / OpenLayers web viewer. |
| **Server** | `py` | Local Python server for serving tiles and styles. |
| **Launcher**| `bat` + `vbs` / `sh` | Script to start the server and launch the browser. |

**_*Only when marker symbols being used in the project_**

## Use Cases

* **Client Side Maps** — build client-side maps using **[MapLibre](https://maplibre.org/)**, **[OpenLayers](https://openlayers.org/)**, **[MapTiler](https://www.maptiler.com/)** or **[Mapbox](https://www.mapbox.com/)**.  
* **OGC Map Services** — publish standard web map services via **[GeoServer*](https://geoserver.org/)** or **[QGIS Server](https://qgis.org/)**.  
* **Project Distribution** — package projects into a single styled layer and a single data source.  

**_*Using the **[MBTiles](https://docs.geoserver.org/main/en/user/community/mbtiles/)** and **[MBStyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/)** extensions_**

## Support

Encounter an error or QGIS–MapLibre mismatch? Please **[open an issue](https://github.com/GallPeters/QGIS2VectorTiles/issues)**, and I will do my best to help.

Follow updates and releases on my **[Mastodon](https://mastodon.social/@JossefKanter)** account.
