<div align="center">

<img width="90" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

# QGIS2VectorTiles

[![Issues](https://img.shields.io/badge/Issues-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-GPL--2.0-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)

<h3>From styled QGIS projects to ready-to-use web maps in a single click</h3>

<kbd><img width="750" alt="QGIS2VectorTilesDemo" src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" /></kbd>

</div>

## Introduction

**QGIS2VectorTiles** is a QGIS plugin that exports projects into web-ready vector tile packages.

It enables users to design maps using QGIS Desktop and package them into a light web map in a single click.

> **Installation:** Available via the official **[QGIS Plugin Repository](https://plugins.qgis.org/)** or the built-in QGIS **Plugin Manager**.

### Key Features
* **Design process** - Native support for the QGIS graphical interface, flexible expressions, geometry generators, and a variety of data formats.
* **Output Dataset** - Vector tiles generated via the GDAL MBTiles driver.
* **Output Style** - MapLibre-compliant web style matching the original QGIS cartography as closely as possible.
* **Tight Data-Styling Coupling** - Client only receives the specific data and styling rules required for rendering.

## Generated Package

| Component | Format | Description |
| :--- | :--- | :--- |
| **Tiles** | `mbtiles` | A single vector tile dataset. |
| **Style** | `json` | MapLibre style sheet matching the original QGIS project design. |
| **Sprites** | `png` + `json` | Marker symbol package generated only when required. |
| **Viewer** | `html` + `js` + `css` | Offline MapLibre and OpenLayers web viewer. |
| **Server** | `py` | Local Python server for serving tiles and styles. |
| **Launcher**| `bat` + `vbs` / `sh` | Platform-specific scripts to start the server and launch the browser. |

## Key Use Cases

* **Web Mapping Applications** - Build client-side maps using **[MapLibre](https://maplibre.org/)**, **[OpenLayers](https://openlayers.org/)**, **[MapTiler](https://www.maptiler.com/)**, or **[Mapbox](https://www.mapbox.com/)**.
* **OGC Map Services** - Publish maps using standard WMS/WMTS protocols via **[GeoServer](https://geoserver.org/)** (using the **[MBTiles](https://docs.geoserver.org/main/en/user/community/mbtiles/)** and **[MBStyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/)** extensions) or via **[QGIS Server](https://qgis.org/)**.
* **Project Distribution** - Package cartographic projects into a single styled layer and a single data source.

## Support & Community

Encounter an error or QGIS-MapLibre mismatch? Please **[open an issue](https://github.com/GallPeters/QGIS2VectorTiles/issues)** and I will do my best to help.

Follow updates and releases on my **[Mastodon](https://mastodon.social/@JossefKanter)** account.
