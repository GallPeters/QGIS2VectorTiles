# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) that packages a QGIS project into:  
- **Vector tiles source** (XYZ directory)  
- **Vector tiles layer** (.qlr)  
- **Client-side MapLibre style package** (style.json)  

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

| Feature | Limitation | Workaround |
|---|---|---|
| Styling | MapLibre more limited than QGIS | Use simple styling |
| Polygon labels | Centroid only | Avoid perimeter/multipart reliance |
| Colors | RGB only | No opacity/transparency |
| Polygon outlines | Not supported | Add simple-line layer |
| Geometry generators | Converted to geometry | Prefer generators over unique QGIS-only styling |
| Marker symbols | Rasterized; DDP ignored | Avoid data-driven markers for HTML |
| Fonts | HTML uses local fonts | Use MapLibre Font Maker |
| Renderers & labels | Converted to rule-based; blocking labels ignored | Avoid blocking-label configs |
| Viewer | Must serve tiles/styles | Run `serve.bat` |

---

## How It Works
1. Converts renderers/labels to rule-based style.  
2. Flattens nested rules with property inheritance (filters, zoom, symbol layers).  
3. Splits rules by zoom, symbol layers, and label rules.  
4. Exports datasets with calculated fields, geometry transforms, and centroids.  
5. Generates vector tiles via GDAL MVT driver.  
6. Loads tiles back into QGIS as QgsVectorTileLayer.  
7. Exports client-side style.json with sprites for markers/labels.  

---

## Tiling Scheme
| Zoom | Scale |
|:-:|:-:|
|0–22|419,311,712 → 99 (ref. scale)|  

---

## Changelog
**v1.7 (16.02.26)**: DDP as fields, removed MBTiles, improved docs & demos  
**v1.6 (14.02.26)**: Experimental MapLibre style output, bug fixes  
**v1.5 (08.02.26)**: Fields-based properties, faster processing  
**v1.4–v1.0**: Various improvements, QT6 support, initial release  

---

## License
GNU GPL v2 → [LICENSE](https://www.gnu.org/licenses/gpl-2.0-standalone.html)  

---

## Contributing
Open an issue or submit a pull request.
