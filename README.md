# <img align="center" width="45" height="45" alt="QGIS2VectorTiles Icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles


**QGIS2VectorTiles** is a QGIS Plugin designed to pack a QGIS projects into a self-contained vector tiles package while preserving original styling.

The plugin, which is avaliable in the [QGIS Plugin Repository](https://plugins.qgis.org/plugins/QGIS2VectorTiles/), generate a ready-to-use package which includes:
- **Vector Tiles:** `XYZ directory`
- **Styled QGIS Layer:** `.qlr`
- **MapLibre Style:** `.json`
- **MapLibre Sprites:** `.png` & `.json`
- **MapLibre Viewer:** `.html` & `.bat` / `.sh`

## Demos

Cool demos are avaliable in the [demos page](https://github.com/GallPeters/QGIS2VectorTiles/tree/main/assets/demos).

## Changelog

A noteblae changes list is avaliable in the [changelog page](https://github.com/GallPeters/QGIS2VectorTiles/blob/main/CHANGELOG.md)

## Use Cases
1. **Client-Side Rendering**  
   Generate client-side compatible styles for web clients like `MapLibre`, `OpenLayers`, `Leaflet` (using plugin), `MapTiler` and `Mapbox`.  

2. **Server-side rendering**  
   Serve lightweight vector tiles via `QGIS Server` while keeping QGIS advanced cartogrpahy or use client-style with `GeoServer` (using plugin).

3. **Project Sharing**  
   Package big projects with big number of layers and data-sources (PostGIS, GeoPackage, shapefiles) styled using complex cartography (variety of renderers and labeling types, data-driven properties, geometry generators and more) into a single data-source and a single styled layer file.  

---

## Styling Considerations

| Title | Limitation / Behavior | How to handle |
|---|---|---|
| Styling Support | MapLibre styling is more limited than QGIS styling. | Use simple styling and avoid complex QGIS-only features. |
| Polygon Labels | Labels use single-part centroids. No perimeter labels. Multipart polygons → multiple labels. | Design labels for centroid placement only. |
| Colors | RGB only. No alpha/transparency. | Avoid opacity and transparency. |
| Polygon Outlines | Polygon outlines are not supported. | Add a simple-line symbol layer instead. |
| Geometry Generators | Converted to resulting geometry. | Prefer geometry generators over unique QGIS styling options. |
| Marker Symbols | Converted to raster icons. Data-defined properties not shown in HTML. | Do not rely on data-defined marker properties for HTML output. |
| Fonts | Glyphs are not being served. HTML uses local fonts. | Generate fonts with [MapLibre Font Maker](https://maplibre.org/font-maker/) and update the style. |
| Renderers & Labels | Are being converted to rule-based. Blocking labels not supported. | Avoid blocking-label configurations. |
| Styling Accuracy | Complex styles may not fully match QGIS output. | Prefer expression-based styling (geometry generators, data-defined). |
| Viewer Usage | Tiles and style must be served to display. | Run `serve.bat` from the output folder. |

---
## How It Works

1. **Renderer and labeling conversion**: Converts all renderers and labelings to a rule-based type for consistency.  
2. **Flattening nested rules**: Nested rules are flattened into standalone rules with inherited properties, including:  
   - Filter expressions  
   - Zoom level ranges  
   - Symbol layers (for renderer rules)  
3. **Rule splitting**: Rules are split based on:  
   - Zoom levels (if `@map_scale` variable is used)  
   - Symbol layers (for renderer rules)  
   - Matching renderer rules (for labeling rules)  
4. **Dataset export**: Each rule becomes a separate dataset:  
   - Calculates expression-based fields (label expressions, data-driven properties, etc.)  
   - Transforms geometry where needed (geometry generators, polygon centroids for labels and fill layers, line conversion for polygons outlines)  
5. **Vector tile generation**: Uses the fast GDAL MVT driver to produce vector tiles efficiently.  
6. **Loading into QGIS**: Styled tiles are loaded back as `QgsVectorTileLayer`.
7. **Client-side style export**: Generates a MapLibre-compatible `style.json` with sprites for markers symbols (inside renderers and labeling backgrounds) ensuring the tiles are ready for web clients.  

---

## Tiling Scheme
| Zoom |  Reference Scale |
|:-:|:-:|
|0 → 22|419,311,712 → 99|  

---

## License
[GNU GPL v2](https://www.gnu.org/licenses/gpl-2.0-standalone.html)  

---

## Contributing
Contributions welcome! Please open an issue or submit a pull request.