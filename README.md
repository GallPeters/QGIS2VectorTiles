# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) that packages a QGIS project into:  
- **Vector tiles source** (XYZ directory)  
- **Vector tiles layer** (.qlr)  
- **MapLibre style package** (style.json and sprites)  

---
## Demos
- **Europe (QGIS built-in world)**: zoom 0–5, ~20 sec 
  
  https://github.com/user-attachments/assets/8f057667-7fd1-4062-bfcd-79f8a09f2118


- **USA (Natural Earth project)**: zoom 0–7, ~13 min 
 
  https://github.com/user-attachments/assets/9fcc00af-f729-4ced-be85-8f7ba07e7eff 

---

## Use Cases
1. **Client-side Rendering (Experimental)**  
   Generate MapLibre/Mapbox-compatible styles for web clients. *Note: Not all QGIS styling supported.*  

2. **Efficient WMS/WMTS Serving**  
   Serve lightweight vector tiles via QGIS Server while keeping QGIS cartography features.  

3. **Simple Project Sharing**  
   Package complex projects (PostGIS, GeoPackage, shapefiles) into a single XYZ source + one layer file.  

---

## Key Considerations

| Feature | Limitation | Workaround / Notes |
|---|---|---|
| MapLibre styling | Client-side style spec is limited compared to QGIS | Not all QGIS styling features are supported; simplify complex styles |
| Polygon labels | Converted to single-part centroids | Perimeter labels not supported; multipart polygons produce multiple labels |
| Colors | Only RGB supported | Avoid alpha/transparency channels |
| Polygon outlines | Not supported | Use simple-line symbol layer to simulate outlines |
| Geometry generators | Converted to resulting geometry | Prefer geometry generators over QGIS-only unique styling options |
| Marker symbols | Rasterized; DDP ignored in HTML | Data-defined properties visible in QGIS output only |
| Fonts | Not compressed/served; HTML uses local fonts | Use [MapLibre Font Maker](https://maplibre.org/font-maker/) and reference fonts in style |
| Renderers & labels | Converted to rule-based | Blocking labels and complex label settings are not supported |
| Performance notes | Complex QGIS styles may not fully match MapLibre rendering | Use simpler or expression-based styling where possible |

---


## How It Works

1. **Renderer and labeling conversion**: Converts all renderers and labelings to a rule-based type for consistency.  
2. **Flattening nested rules**: Nested rules are flattened into standalone rules with inherited properties, including:  
   - Filter expressions  
   - Zoom level ranges  
   - Symbol layers for renderer rules  
3. **Rule splitting**: Rules are split based on:  
   - Zoom levels (if `@map_scale` variable is used)  
   - Symbol layers (renderer rules)  
   - Matching renderer rules (labeling rules)  
4. **Dataset export**: Each rule becomes a separate dataset:  
   - Calculates expression-based fields (label expressions, DDP, etc.)  
   - Transforms geometry where needed (geometry generators, polygon centroids for labels/fill layers, line conversion for polygons)  
5. **Vector tile generation**: Uses the fast GDAL MVT driver to produce vector tiles efficiently.  
6. **Loading into QGIS**: Styled tiles are loaded back as `QgsVectorTileLayer` for preview and validation.  
7. **Client-side style export**: Generates a MapLibre-compatible `style.json` with sprites for markers and label backgrounds, ensuring the tiles are ready for web clients.  
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

## Tiling Scheme
| Zoom | Scale |
|:-:|:-:|
|0–22|419,311,712 → 99 (ref. scale)|  

---

## License
[GNU GPL v2](https://www.gnu.org/licenses/gpl-2.0-standalone.html)  

---

## Contributing
Open an issue or submit a pull request.
