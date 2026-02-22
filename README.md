# <img align="center" width="45" height="45" alt="QGIS2VectorTiles Icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles


**QGIS2VectorTiles** is a QGIS Plugin designed to package your QGIS projects into self-contained vector tile package while preserving your original styling.

>Avaliable on the [QGIS Plugin Repository](https://plugins.qgis.org/plugins/QGIS2VectorTiles/)

Each export automatically generates a complete, deployment-ready package:
- **Vector Tiles:** `XYZ directory`
- **Styled QGIS Layer:** `.qlr`
- **MapLibre Style:** `.json`
- **MapLibre Sprites:** `.png` & `.json`
- **MapLibre Viewer:** `.html` & `.bat` / `.sh`

---

## Use Cases

<img style="width:90%" alt="QGIS2VectorTilesUseCases" src="https://github.com/user-attachments/assets/f2c53e54-e887-4537-8591-20f72b5458df" />

---

## Workflow

<img style="width:90%" alt="QGIS2VectorTiles Workflow" src="https://github.com/GallPeters/QGIS2VectorTiles/blob/main/assets/QGIS2VectorTilesWorkflow.png"/>

---

## Demos
The demo below show the convertion of the [Natural Earth Project](https://www.naturalearthdata.com/) which contains 235 layers in the USA area (zoom 0–7) within 13 minutes (More examples are avaliable in the [Demos Page](https://github.com/GallPeters/QGIS2VectorTiles/blob/main/assets/demos/Demos.md)).

https://github.com/user-attachments/assets/6b786617-7294-4fb5-a884-42e2bc0cf2e2

---

## Technical Specifications

| Topic | Plugin Behavior | Recommendation |
|---|---|---|
| **Zoom levels** | QGIS Desktop tiling scheme is used. |  Use predfined scales (z0 → 1:419,311,712, z22 → 1:99) 
| **Styling Support** | MapLibre supports a narrower range of styling options compared to QGIS. | Utilize simple styles and avoid QGIS-exclusive rendering features. |
| **Polygon Labels** | Labels are anchored to single-part centroids. Multipart polygons will generate multiple labels. | Design and position labels exclusively for centroid placement. |
| **Color Profiles** | Restricted to RGB. Alpha channels and transparency are not supported. | Use solid colors; avoid opacity and transparency settings. |
| **Polygon Outlines** | Native polygon outlines are not supported. | Apply a simple-line symbol layer to represent outlines. |
| **Geometry Generators** | Automatically converted to their resulting static geometry. | Rely on geometry generators rather than unique QGIS styling properties. |
| **Marker Symbols** | Converted to static raster icons. Data-defined properties will not translate to the HTML output. | Avoid using data-defined marker properties if relying on the HTML viewer. |
| **Typography** | Glyphs are not automatically served. HTML output defaults to local system fonts. | Generate custom fonts using [MapLibre Font Maker](https://maplibre.org/font-maker/) and update the output style accordingly. |
| **Renderers & Labels** | Converted to rule-based rendering. Blocking labels are unsupported. | Remove blocking-label configurations prior to export. |
| **Viewer Usage** | The HTML viewer requires tiles and styles to be actively served. | Execute the `serve.bat` or `serve.sh` script located in the output directory. |

---

## Changelog

Please refer to the [CHANGELOG.md](https://github.com/GallPeters/QGIS2VectorTiles/blob/main/CHANGELOG.md) for a detailed history of updates and releases.

---

## Contributing

We welcome community contributions! Please feel free to open an issue to report bugs or submit a pull request with improvements.

---

## License

This project is licensed under the [GNU GPL v2](https://www.gnu.org/licenses/gpl-2.0-standalone.html).
