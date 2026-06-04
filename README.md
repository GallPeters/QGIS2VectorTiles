<div align="center">



<img width="90" alt="icon" src="https://github.com/user-attachments/assets/e1f0e64b-6850-4ae5-b3c0-2ce2fca5580e" />

  # QGIS2VectorTiles



[![Issues](https://img.shields.io/badge/Issues-🛠️-98b023?style=for-the-badge)](https://github.com/GallPeters/QGIS2VectorTiles/issues)
[![Homepage](https://img.shields.io/badge/Homepage-🚀-black?style=for-the-badge)](https://gallpeters.github.io/QGIS2VectorTiles/)
[![License](https://img.shields.io/badge/License-📜-98b023?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html)


 **From styled QGIS projects to client-side web maps in just one click.**

<kbd> <img width="500" alt="QGIS2VectorTilesDemo"  src="https://github.com/user-attachments/assets/98da33f5-7513-4f84-a8a7-4d0750d7db63" /></kdb>

</div>



### Introduction

QGIS2VectorTiles generates a vector tile package directly from the current QGIS project.

The plugin is designed to bridge the gap between the rich and highly flexible environment of **QGIS** Desktop - supporting complex spatial expressions, a wide range of data formats, geometry generators, and data-driven styling - and modern web mapping frameworks, which rely on client-side rendering of large datasets.

The primary target style is **MapLibre** style specification and compatible web mapping libraries.
To ensure portability and ease of installation, the tile generation process is based on the built-in **GDAL** MBTiles driver, avoiding external dependencies such as Tippecanoe. While Tippecanoe may offer higher performance in some scenarios, it requires separate installation and is primarily limited to Linux environments

### Output Package

| Component     | Format         | Description                                                                 |
|---------------|--------------------------|-----------------------------------------------------------------------------|
| **Tiles**         | `mbtiles`      | Vector tile dataset generated from project layers and data sources.        |
| **Style**         | `json`         | Client-side style sheet following the MapLibre style specification. |
| **Sprite***       | `png` + `json` | Icon package containing symbol images and metadata used by the client to resolve icons. |
| **Viewer**        | `html`         | Ready-to-use MapLibre/OpenLayers viewer referencing the tiles and style.   |
| **Server**        | `py`           | Local server that serves tiles and style resources locally.                |
| **Launcher**      | `bat` + `vbs` / `sh` | Platform-specific scripts for starting the local tile server and opening the viewer. |

\* Optional. Generated only when required by the style.

### Use Cases

- **Web mapping applications** - Client-side vector tile rendering using web mapping libraries like **[MapLibre](https://maplibre.org), [OpenLayers](https://openlayers.org), [MapTiler](https://www.maptiler.com/), [Mapbox](https://www.mapbox.com/)** and more.

- **Map services** - WMS and WMTS publishing using standard map server like [**Geoserver**](https://geoserver.org/) (using [mbtiles](https://docs.geoserver.org/main/en/user/community/mbtiles/) and [mbstyle](https://docs.geoserver.org/main/en/user/styling/mbstyle/) extentions) and [**QGIS Server**](https://qgis.org/)

- **Projects distribution** - Distribution of complex cartographic outputs with a single styled layer and a single source. 
