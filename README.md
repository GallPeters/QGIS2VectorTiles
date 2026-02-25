# <img align="center" width="45" height="45" alt="QGIS2VectorTiles Icon" src="https://github.com/user-attachments/assets/0080b326-2fa3-4c42-b096-a946cf77a69c" /> QGIS2VectorTiles

**QGIS2VectorTiles** packs a QGIS project into a fully styled vector tile package ready for MapLibre. It allows users to design maps in the familiar and flexible QGIS Desktop and export stunning MapLibre client-side web maps using GDAL’s powerful MVT driver.

The output package includes:
- Vector tile source (`XYZ` directory)
- Styled QGIS layer file (`.qlr`)
- MapLibre style (`.json`)
- MapLibre sprites *(optional)* (`.png` + `.json`)
- A ready-to-use MapLibre viewer (`.html`)

This package is ideal for:
- **Client-side web map rendering** — MapLibre, OpenLayers, and Leaflet
- **Lightweight WMS/WMTS vector tile serving** — GeoServer and QGIS Server
- **Sharing complex, multi-source styled QGIS projects** — packaged as a single data source and layer file

More information is available on the [plugin homepage](https://gallpeters.github.io/QGIS2VectorTiles/).