# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) that packages a QGIS project into vector tiles package while preserving original style. The package includes:  
- **Vector Tiles** (XYZ directory)  
- **Styled QGIS Vector Tiles Layer** (.qlr)  
- **MapLibre Style Document** (.json)  
- **MapLibre Sprites** (.png & .json)
- **MapLibre Viewer** (.html)
- **Viewer Launcher** (.bat / .sh)

---

## Use Cases
1. **Client-Side Rendering (Experimental)**  
   Generate MapLibre compatible styles for web clients. *Note: Not all QGIS styling supported.*  

2. **Efficient WMS/WMTS Serving**  
   Serve lightweight vector tiles via QGIS Server while keeping QGIS advanced cartography.  

3. **Project Sharing**  
   Package complex projects (PostGIS, GeoPackage, shapefiles) into a single source and a single layer file.  

---
## Demos
- _**Basic** - QGIS built-in world dataset (1 layer) in Europe area (zoom 0–5) &rarr; 20 seconds:_
  
  https://github.com/user-attachments/assets/8f057667-7fd1-4062-bfcd-79f8a09f2118


- _**Complex** - [Natural Erath Project](https://www.naturalearthdata.com/) (235 layers) in USA area (zoom 0–7) &rarr; 13 minutes:_
 
  https://github.com/user-attachments/assets/9fcc00af-f729-4ced-be85-8f7ba07e7eff 

- _**Data Driven** - QGIS built-in world dataset (1 layer) in Europe area (zoom 0–5) &rarr; 10 seconds:_

  https://github.com/user-attachments/assets/61e46690-e0ab-4c72-8f11-c2350381e9f8

---

## Tiling Scheme
| Zoom |  Reference Scale |
|:-:|:-:|
|0 → 22|419,311,712 → 99|  

---

## Changelog
[Changelog](https://github.com/GallPeters/QGIS2VectorTiles/blob/main/CHANGELOG.md)

---


## Key Considerations

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
1. **Convert To Rule-Based**: Converts all renderers and labelings to a rule-based type.
2. **Flatten Rules**: Flattens rules  into standalone rules while inheritance:  
   - Filter expressions.  
   - Zoom ranges.
   - Symbol layers (for renderer rules).
3. **Split Rules**: Splits Rules based on:  
   - Zoom levels (for rules uses @map_scale)  
   - Symbol layers (for renderer rules)  
   - Matching renderers rules (for labeling rules)  
4. **Export Rules To Datasets**: Exports each rule to a seperated datasets while:  
   - Calculates expression-based fields (label expressions, data-driven properties, etc.)  
   - Transforms geometry where needed (geometry generators, polygons labels into centroids, polygons outlines into lines)  
5. **Generate Vector Tiles**: Generates `XYZ Directory` contains vector tiles using efficiently GDAL's MVT driver.
6. **Load Styled Tiles Into QGIS**: Loads styled tiles back into QGIS as vector tiles layer and export it into `.qlr`.
7. **Produce MapLibre Style from Tiles Layer**: Produce a MapLibre-compatible style includes:
   -  Style `.json` document.
   -  Sprite package include `.png` and `.style` files (for renderers and labeling backgrounds contains marker symbols).
8. **Copy launcher** Copy ready-to-use launcher includes:
   - MapLibre `.html` viewer.
   - Launcher `.bat` file which serves tiles and launch viewer.

---



## License
[GNU GPL v2](https://www.gnu.org/licenses/gpl-2.0-standalone.html)  

---

## Contributing
Contributions welcome! Please open an issue or submit a pull request.
