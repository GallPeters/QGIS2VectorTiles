<div align="center">



<img width="90" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

  # QGIS2VectorTiles



[![Issues](https://img.shields.io/badge/Issues-🛠️-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-🚀-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)


 **From styled QGIS projects to client-side web maps in just one click.**

<kbd> <img width="500" alt="QGIS2VectorTilesDemo"  src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" /></kdb>

</div>


## Introduction

> **QGIS2VectorTiles** generates a complete vector tile package directly from the current QGIS project.

The output includes a vector tile dataset, a client-side style package that preserves the original QGIS cartography as closely as possible, and a ready-to-use tile server and web viewer. This enables QGIS Desktop maps to be exported to the web for offline use or online publishing in a single click.

## Background

> **QGIS** (cartographic design) → **GDAL** (tile generation) → **MapLibre** (web rendering)

[**MapLibre**](https://maplibre.org/) provide fast client-side rendering of large spatial datasets, making them ideal for interactive web maps. However, their cartographic capabilities are intentionally more limited than those available in desktop GIS software.

[**QGIS**](https://www.qgis.org/), offers a rich cartographic environment with advanced symbology, expressions, geometry generators, labeling engines, and support for a wide range of spatial data formats. While this makes QGIS an excellent platform for map design, it is not intended to serve web maps directly.

[**GDAL**](https://gdal.org/en/stable/) bridges these two environments by providing a powerful and efficient vector tile generation engine. However, GDAL workflows are primarily command-line based and require additional configuration to connect desktop cartography with modern web mapping frameworks.

**QGIS2VectorTiles** brings these technologies together into a single workflow. It allows users to design and style maps in QGIS, generate vector tiles using GDAL, and publish them as fast client-side web maps using MapLibre or OpenLayers - all from a familiar desktop interface.


## Use Cases

- **Web mapping applications** - Client-side vector tile rendering using web mapping libraries like **[MapLibre](https://maplibre.org), [OpenLayers](https://openlayers.org), [MapTiler](https://www.maptiler.com/), [Mapbox](https://www.mapbox.com/)** and more.

- **Map services** - WMS and WMTS publishing using standard map server like [**Geoserver**](https://geoserver.org/) (using [**MBTiles**](https://docs.geoserver.org/main/en/user/community/mbtiles/) and [**MBStyle**](https://docs.geoserver.org/main/en/user/styling/mbstyle/) extentions) and [**QGIS Server**](https://qgis.org/)

- **Projects distribution** - Distribution of complex cartographic outputs with a single styled layer and a single source. 


## Output Packag

```
root/
├── tiles.mbtiles
│
├── style/
│   ├── style.json
│   ├── tiles.qlr
│   │
│   └── sprite/                  # Optional (only if marker symbols are used)
│       ├── sprite.png
│       ├── sprite.json
│       ├── sprite@2x.png
│       └── sprite@2x.json
│
├── utils/
│   ├── tiles_server.py
│   │
│   ├── viewer/
│   │   ├── viewer.html
│   │   ├── maplibre.js        # Can be OpenLayers (depends on export)
│   │   └── maplibre.css       # Can be OpenLayers (depends on export)
│   │
│   ├── activate_server.bat     # Windows (optional)
│
├── activate_server.vbs         # Windows shortcut (optional)
└── activate_server.sh          # Linux/macOS (optional)
```
