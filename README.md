# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) that packages a QGIS project into vector tiles package while preserving original style:  
- **Vector Tiles** (XYZ directory)  
- **Styled QGIS Vector Tiles Layer** (.qlr)  
- **MapLibre Style Package** (style.json and sprites)  

---
## Demos
- _**Basic** - QGIS built-in world dataset (1 layer) in Europe area (zoom 0–5) &rarr; 20 seconds:_
  
  https://github.com/user-attachments/assets/8f057667-7fd1-4062-bfcd-79f8a09f2118


- _**Complex** - [Natural Erath Project](https://www.naturalearthdata.com/) (235 layers) in USA area (zoom 0–7) &rarr; 13 minutes:_
 
  https://github.com/user-attachments/assets/9fcc00af-f729-4ced-be85-8f7ba07e7eff 

- _**Data Driven** - QGIS built-in world dataset (1 layer) in Europe area (zoom 0–5) &rarr; 10 seconds:_

  https://github.com/user-attachments/assets/61e46690-e0ab-4c72-8f11-c2350381e9f8


---

## Use Cases
1. **Client-Side Rendering (Experimental)**  
   Generate MapLibre compatible styles for web clients. *Note: Not all QGIS styling supported.*  

2. **Efficient WMS/WMTS Serving**  
   Serve lightweight vector tiles via QGIS Server while keeping QGIS advanced cartogrpahy.  

3. **Project Sharing**  
   Package complex projects (PostGIS, GeoPackage, shapefiles) into a single source and a single layer file.  

---

## Key Considerations
## Considerations

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

## Changelog

**v1.7 (16.02.26)**  
- Implemented Data Defined Properties (DDP) as fields in MapLibre style.json.  
- Removed MBTiles output (not supported by native MapLibre).  
- Improved documentation, added more demos and limitations guidance.  

**v1.6 (14.02.26)**  
- Experimental client-side style generation: outputs MapLibre/Mapbox style.json + sprites.  
- Bug fixes for smoother style conversion, especially with geometry generators and centroid fill symbol layers.  

**v1.5 (08.02.26)**  
- Added support for fields-based properties alongside expression-based ones.  
- Processing performance improvements.  

**v1.4 (08.02.26)**  
- Improved rendering for Centroid Fill symbol layers.  
- Stabilized DDP support on QGIS LTR (3.40.x).  

**v1.3 (05.02.26)**  
- Added QT6 support.  
- Improved output rendering by matching calculated field types of DDP to property types.  
- Performance improvements: skipped unnecessary rule splitting/exporting steps.  
- Bug fixes preventing crashes when retrieving specific DDP on QGIS LTR.  

**v1.2 (30.12.25)**  
- Added “Polygons Labels Base” parameter to processing.  

**v1.1 (16.12.25)**  
- Supported DDP as calculated fields for better rendering performance.  
- Minor bug fixes.  

**v1.0 (10.12.25)**  
- Initial release.  

---

## License
[GNU GPL v2](https://www.gnu.org/licenses/gpl-2.0-standalone.html)  

---

## Contributing
Contributions welcome! Please open an issue or submit a pull request.
