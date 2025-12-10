# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" />      QGIS2VectorTiles
 
Pack a QGIS project into a single vector tiles source (.mbtiles or XYZ) and a single vector tiles layer (.qlr).

## Demo

*Converting the [Natural Earth quick-start project](https://www.naturalearthdata.com/) (USA area) to vector tiles in zoom levels 0-8 (sped up from 8 minutes)*


https://github.com/user-attachments/assets/13d54e68-f5ca-46a1-866f-c93a2a9c5d88


## Use Cases

### 1. Efficient WMS/WMTS Serving
Serve lightweight vector tiles via QGIS Server instead of heavy raster tiles. Leverage QGIS's full cartographic capabilities (multiple renderer types, polygons labels and outlines,complex expressions based properties, geometry generators) which are not available in client-side vector tile specs.

### 2. Easy Project Sharing
Replace big, messy and complex projects with multiple layers and data sources (PostGIS, GeoPackage, shapefiles, etc.) with:
- 1 layer file
- 1 data source

### 3. Client-Side Rendering (In Development)
Generate client-side (MapLibre) compatible style:
- style
- sprites
- glyphs

## How It Works

1. Converts renderers and labelings to a rule-based type
2. Flattens nested rules with property inheritance
3. Splits rules by zoom levels, symbol layers and match renderer rules 
4. Exports each rule as separate dataset with the required geometry transformations
5. Generates vector tiles using GDAL MVT driver
6. Loads styled tiles back into QGIS

## License

This project is licensed under the GNU General Public License v3.0.

See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0-standalone.html) file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
