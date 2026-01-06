# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" />      QGIS2VectorTiles
 
**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) which pack a QGIS project into a single vector tiles source (.mbtiles or XYZ) and a single vector tiles layer (.qlr).

## Demo

*1. Converting the [Natural Earth quick-start project](https://www.naturalearthdata.com/) (USA) to vector tiles in zoom levels 0-8 (sped up from 8 minutes)*

https://github.com/user-attachments/assets/13d54e68-f5ca-46a1-866f-c93a2a9c5d88

*2. Converting the [Natural Earth quick-start project](https://www.naturalearthdata.com/) (part of Europe) to vector tiles in zoom levels 0-8 (sped up from 6 minutes)*

https://github.com/user-attachments/assets/6457b05a-f00a-4935-bea1-28b0d8550e8f


## Use Cases

### 1. Efficient WMS/WMTS Serving
Serve lightweight vector tiles via QGIS Server instead of heavy raster tiles. Leverage QGIS's full cartographic capabilities (multiple renderer types, polygons labels and outlines,complex expressions based properties, geometry generators) which are not available in client-side vector tile specs.

### 2. Easy Project Sharing
Replace big, messy and complex projects with multiple layers and data sources (PostGIS, GeoPackage, shapefiles, etc.) with a single data source (.mbtiles or XYZ) and a single layer file (.qlr).

### 3. Client-Side Rendering (In Development)
Generate a client-side (MapLibre) compatible style:
- sprites
- style
- glyphs

## How It Works

1. Converts renderers and labelings to a rule-based type
2. Flattens nested rules with property inheritance
3. Splits rules by zoom levels, symbol layers and match renderer rules 
4. Exports each rule as separate dataset with the required geometry transformations
5. Generates vector tiles using GDAL MVT driver
6. Loads styled tiles back into QGIS

## Tiling scheme
| Zoom Level | Reference Scale |
| :-: | :-: |
|0|419311712|
|1|209655856|
|2|104827928|
|3|52413964|
|4|26206982|
|5|13103491|
|6|6551745|
|7|3275872|
|8|1637936|
|9|818968|
|10|409484|
|11|204742|
|12|102371|
|13|51185|
|14|25592|
|15|12796|
|16|6398|
|17|3199|
|18|1599|
|19|799|
|20|399|
|21|199|
|22|99|

## Changelog
### v1.2 (30.12.25):
- add Polygons Labels Base parameter
### v1.1 (16.12.25):
- Fix bugs
- Support expressions based properties
### v1.0 (10.12.25):
- Initial
## License

This project is licensed under the GNU General Public License v3.0.

See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0-standalone.html) file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
