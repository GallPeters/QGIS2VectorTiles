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

QGIS2VectorTiles generates a vector tile package directly from the current QGIS project.

The plugin combines the strengths of [**QGIS**](https://www.qgis.org/), [**GDAL**](https://gdal.org/en/stable/), and [**MapLibre**](https://maplibre.org/), allowing rich desktop cartography to be transformed into fast, client-side web maps in a single click.

> **QGIS** (cartographic design) → **GDAL** (tiles generation) → **MapLibre** (web rendering)


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
## Use Cases

- **Web mapping applications** - Client-side vector tile rendering using web mapping libraries like **[MapLibre](https://maplibre.org), [OpenLayers](https://openlayers.org), [MapTiler](https://www.maptiler.com/), [Mapbox](https://www.mapbox.com/)** and more.

- **Map services** - WMS and WMTS publishing using standard map server like [**Geoserver**](https://geoserver.org/) (using [mbtiles](https://docs.geoserver.org/main/en/user/community/mbtiles/) and [mbstyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/) extentions) and [**QGIS Server**](https://qgis.org/)

- **Projects distribution** - Distribution of complex cartographic outputs with a single styled layer and a single source. 
