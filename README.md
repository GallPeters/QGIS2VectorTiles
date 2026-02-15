# <img align="center" width="45" height="45" alt="icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" />      QGIS2VectorTiles
 
**QGIS2VectorTiles** is a [QGIS Plugin](https://plugins.qgis.org/plugins/QGIS2VectorTiles/) which pack a QGIS project into a single vector tiles source (.mbtiles or XYZ), a single vector tiles layer (.qlr) and a client-side compatible Maplibre/Mapbox style package (style.json and matching sprites).

## Demo

*Converting the [Natural Earth quick-start project](https://www.naturalearthdata.com/) (part of Europe) to vector tiles in zoom levels 0-8 (sped up from 6 minutes) then serve it from a QGIS Server instance to an OpenLayers viewer.*

https://github.com/user-attachments/assets/6457b05a-f00a-4935-bea1-28b0d8550e8f


## Use Cases

### 1. Efficient WMS/WMTS Serving
Serve lightweight vector tiles via QGIS Server instead of heavy raster tiles. Leverage QGIS's full cartographic capabilities (multiple renderer types, polygons labels and outlines,complex expressions based properties, geometry generators) which are not available in client-side vector tile specs.

### 2. Easy Project Sharing
Replace big, messy and complex projects with multiple layers and data sources (PostGIS, GeoPackage, shapefiles, etc.) with a single data source (.mbtiles or XYZ) and a single layer file (.qlr).

### 3. Client-Side Rendering (Experimental)
Generate a MapLibre/Mapbox client-side compatible style which can be used with Maplibre, Mapbox, OpenLayers, Leaflet, MapTiler, etc. in order to render vector tiles on the client side. This is an experimental feature and may not support all QGIS styling capabilities.

## How It Works

1. Converts renderers and labelings to a rule-based type.
2. Flattens nested rules to stand alone rules using properties inheritance which includes:
   -   Filter expressions.
   -  Zoom levels ranges.
   -  Symbol layers (for renderer rules).
3. Splits rules by:
   -  Zoom levels (if @map_scale varaiable is being used).
   -  Symbol layers (for renderer rules)
   -  Match renderer rules (for labeling rules)
4. Exports each rule as a separate dataset while:
   - Calculating expression based fields (label expressions, data-driven properties etc).
   - Transform geometry if required (e.g. geometry generators, centroids for polygons labels and centroid fill symbol layers, lines for polygons etc.)
5. Generates vector tiles using (amazingly fast) GDAL MVT driver
6. Loads styled tiles back into QGIS as QgsVectorTileLayer.
7. Export QgsVectorTileLayer flat style into client-side compatible style.json while generting sprites for marker symbol layers and labeling markers symbol backgrounds.

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
### v1.6 (14.02.26):
- Support a client-side style package generation which outputs MapLibre/Mapbox style.json and matching sprites (experimental).
- Bug fixes which ensure smooth style conversion process which involves geometry generators and centroid fills symbol layers.
### v1.5 (08.02.26):
- Support fields based properties in addition to expression based properties.
- Improved processing performence.
### v1.4 (08.02.26):
- Improved renderering of Centroid Fill symbol layers.
- Stabilizing DDP support when running on QGIS LTR (3.40.x).
### v1.3 (05.02.26):
- Added QT6 support
- Improved output rendering performance by matching the calculated field type of DDP to the property type.
- Improved processing performence by skipping unnessesary rule splitting and exporting steps.
- Bug fixes including preventing crashes when retrieving specific DDP when running on QGIS LTR (3.40.x).
### v1.2 (30.12.25):
- Added Polygons Labels Base as processing parameter.
### v1.1 (16.12.25):
- Support DDP (data defined properties) as calculated fields for improved rendering performence.
- Bug fixes.
### v1.0 (10.12.25):
- Initial release.
## License

This project is licensed under the GNU General Public License v2.0.

See the [LICENSE](https://www.gnu.org/licenses/gpl-2.0-standalone.html) file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
