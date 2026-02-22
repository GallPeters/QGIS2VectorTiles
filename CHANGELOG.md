# Changelog

**v1.9 (22.02.26)**
- Serve maplibre-gl.js locally for an offline use.
- Bug fixes ensuring smooth convertion of layers contains empty renderers/labelings.

**v1.8 (19.02.26)**
- Activate http server silently when running on windows.

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
