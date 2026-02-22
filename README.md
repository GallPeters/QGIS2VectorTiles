# <img align="center" width="45" height="45" alt="QGIS2VectorTiles Icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) designed to package your QGIS projects into self-contained vector tile bundles while meticulously preserving your original styling.

Each export automatically generates a complete, deployment-ready package:
- **Vector Tiles:** `XYZ directory`
- **Styled QGIS Layer:** `.qlr`
- **MapLibre Style:** `.json`
- **MapLibre Sprites:** `.png` & `.json`
- **MapLibre Viewer:** `.html` & `.bat` / `.sh`

---

## Core Use Cases

- **Client-Side Rendering:** Export MapLibre-compatible styles optimized for web map clients. *(Note: Not all QGIS styling features are natively supported).*
- **Server-Side Rendering:** Serve lightweight vector tiles via QGIS Server (WMS/WMTS) without sacrificing QGIS cartographic quality.
- **Project Consolidation:** Combine complex projects spanning multiple data sources (PostGIS, GeoPackage, shapefiles) into a single, highly portable source and layer file.

---

## How It Works

<img style="width:180" alt="QGIS2VectorTiles Workflow" src="https://github.com/GallPeters/QGIS2VectorTiles/blob/main/assets/QGIS2VectorTilesWorkflow.png"/>

---

## Demos

### Featured Demo: Complex Rendering
*Natural Earth Project (235 layers), USA, zoom 0–7. Processing time: ~13 minutes.*

https://github.com/user-attachments/assets/6b786617-7294-4fb5-a884-42e2bc0cf2e2

> **View More Examples:** Check out our [Demos Page](https://github.com/GallPeters/QGIS2VectorTiles/blob/main/assets/demos/Demos.md) to see Basic, Data-Driven, and Offline rendering examples.

---

## Technical Specifications

### Tiling Scheme

| Zoom Level | Reference Scale |
|:---:|:---:|
| 0 → 22 | 419,311,712 → 99 |

### Styling Considerations & Best Practices

For the best results, adhere to the following guidelines when preparing your QGIS project for export:

| Topic | Plugin Behavior | Recommendation |
|---|---|---|
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
