# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) that packages a QGIS project into a self-contained vector tiles bundle while preserving its original styling.

Each export produces:
- **Vector Tiles** — `XYZ directory`
- **Styled QGIS Layer** — `.qlr`
- **MapLibre Style** — `.json`
- **MapLibre Sprites** — `.png` & `.json`
- **MapLibre Viewer** — `.html` & `.bat` / `.sh`

---

## Use Cases

**Client-Side Rendering** — Export MapLibre-compatible styles for web map clients. Not all QGIS styling options are supported.

**Server-Side Rendering** — Serve lightweight vector tiles via QGIS Server as WMS/WMTS while retaining QGIS cartographic quality.

**Project Sharing** — Consolidate projects that span multiple data sources (PostGIS, GeoPackage, shapefiles) into a single source and layer file.

---

## Demos

_**Basic** — QGIS built-in world dataset (1 layer), Europe, zoom 0–5 → ~20 seconds_



https://github.com/user-attachments/assets/a1c834c8-bf66-4b8d-a405-6ee807b3aa73



_**Complex** — [Natural Earth Project](https://www.naturalearthdata.com/) (235 layers), USA, zoom 0–7 → ~13 minutes_



https://github.com/user-attachments/assets/6b786617-7294-4fb5-a884-42e2bc0cf2e2



_**Data Driven** — QGIS built-in world dataset (1 layer), Europe, zoom 0–5 → ~10 seconds_



https://github.com/user-attachments/assets/3b70761e-d20d-4365-80b3-10fca4f2f60c



_**Offline** — QGIS built-in world dataset (1 layer), Europe, zoom 0–5 → ~10 seconds_



https://github.com/user-attachments/assets/bf0f3010-1549-4ebc-91d3-bd41e1aeddfa



---

## Tiling Scheme

| Zoom | Reference Scale |
|:----:|:---------------:|
| 0 → 22 | 419,311,712 → 99 |

---

## Styling Considerations

| Topic | Behavior | Recommendation |
|---|---|---|
| **Styling Support** | MapLibre supports fewer styling options than QGIS. | Stick to simple styles; avoid QGIS-exclusive features. |
| **Polygon Labels** | Labels are placed at single-part centroids. Multipart polygons produce multiple labels. | Design labels for centroid placement only. |
| **Colors** | RGB only. No alpha or transparency. | Avoid opacity and transparency values. |
| **Polygon Outlines** | Not supported. | Use a simple-line symbol layer instead. |
| **Geometry Generators** | Converted to their resulting geometry. | Prefer geometry generators over unique QGIS styling options. |
| **Marker Symbols** | Converted to raster icons. Data-defined properties are not reflected in HTML output. | Do not rely on data-defined marker properties for the HTML viewer. |
| **Fonts** | Glyphs are not served automatically. HTML output uses local fonts. | Generate fonts with [MapLibre Font Maker](https://maplibre.org/font-maker/) and update the style. |
| **Renderers & Labels** | Converted to rule-based. Blocking labels are not supported. | Avoid blocking-label configurations. |
| **Viewer Usage** | Tiles and style must be served to display correctly. | Run `serve.bat` from the output folder. |

---

## How It Works

<img width="1408" height="768" alt="QGIS2VectorTiles Workflow" src="https://github.com/GallPeters/QGIS2VectorTiles/blob/main/assets/QGIS2VectorTilesWorkflow.png"/>

---

## Changelog

See [CHANGELOG.md](https://github.com/GallPeters/QGIS2VectorTiles/blob/main/CHANGELOG.md) for release history.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

---

## License

[GNU GPL v2](https://www.gnu.org/licenses/gpl-2.0-standalone.html)
