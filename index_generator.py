"""
QGIS Processing Plugin Wrapper for MBTiles Generation
Generates styled MBTiles from project layers with identical styling
"""

from qgis.core import *


class IndexGenerator:
    def __init__(self, writer:QgsVectorTileWriter, layers:list[QgsVectorLayer], max_vertices: int):
        self.writer = writer
        self.layers = layers
        self.max_vertices= max_vertices
        self.vertices = []
    
    def generate(self):
        pass


    def get_vertices(self):
        for layer in self.layers:
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom:
                    self.vertices.extend(list(geom.vertices()))
    
    def get_matrix(self):
        matrix_set = QgsTileMatrixSet()
        
     
