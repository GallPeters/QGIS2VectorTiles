"""_summary_"""

from dataclasses import dataclass, field
from json import dump
from math import sqrt, ceil, prod
from os.path import join
from io import BytesIO
from typing import Optional as O, TypeAlias as TA
from qgis.core import QgsSymbol, QgsRuleBasedRenderer
from PyQt5.QtCore import QSize, QBuffer, QIODevice
from PyQt5.QtGui import QImage
from PIL import Image

Img: TA = Image.Image
MatrixShape: TA = tuple[int, int]
MatrixRow: TA = tuple[int, int, list[Img]]
ImgCoord: TA = tuple[float, float, float, float]
ImgsCoords: TA = dict[str, ImgCoord]


@dataclass
class SymbolImage:
    """_summary_"""

    symbol: QgsSymbol
    name: str
    img: Img = field(init=False)

    def __post_init__(self):
        self.generate_symbol_img()

    def __getattr__(self, name):
        return getattr(self.img, name)

    def generate_symbol_img(self):
        """_summary_"""
        qt_img = self.symbol.asImage(QSize(1000, 1000))
        pil_img = self.qt_img_to_pil(qt_img)
        bbox = pil_img.getbbox()
        cropped_img = pil_img.crop(bbox)
        cropped_img.name = self.name
        self.img = cropped_img

    def qt_img_to_pil(self, qt_img: QImage) -> Img:
        """_summary_"""
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qt_img.save(buffer, "PNG")
        strio = BytesIO()
        strio.write(buffer.data())
        buffer.close()
        strio.seek(0)
        pil_img = Image.open(strio)
        return pil_img


@dataclass
class SpriteMatrix:
    """_summary_"""

    imgs: list[SymbolImage]
    ratio: tuple[int, int] = 3, 4
    pixelspace: int = 20
    shape: MatrixShape = field(init=False)
    imgsmatrix: list[list[Img]] = field(init=False)
    imgsrows: list[MatrixRow] = field(init=False)

    def __post_init__(self):
        """_summary_"""
        self.calculate_shape()
        self.generate_imgs_matrix()
        self.get_matrix_rows()

    def calculate_shape(self):
        """_summary_"""
        imgs_count = len(self.imgs)
        shape = map(lambda x: ceil(sqrt(imgs_count / prod(self.ratio)) * x), self.ratio)
        height, width = tuple(shape)
        if width * (height - 1) >= imgs_count:
            height -= 1
        self.shape = height, width

    def generate_imgs_matrix(self):
        """_summary_"""
        height, width = self.shape
        symbols = list(self.imgs)
        matrix = [symbols[width * num : width * (num + 1)] for num in range(height)]
        self.imgsmatrix = matrix

    def get_matrix_rows(self):
        """_summary_"""
        matrix_rows = []
        for row in self.imgsmatrix:
            width = sum(img.width + self.pixelspace for img in row) - self.pixelspace
            height = max(img.height for img in row) + self.pixelspace
            matrix_rows.append((width, height, row))
        self.imgsrows = matrix_rows


@dataclass
class SpriteImage:
    """_summary_"""

    matrix: SpriteMatrix
    pixelspace: int = 20
    lowerfactor = 2
    img: Img = field(init=False)
    lowerimg: Img = field(init=False)
    imgscoords: ImgsCoords = field(init=False)

    def __post_init__(self):
        """_summary_"""
        self.construct_img()
        self.populate_img()
        self.generate_lowerimg()

    def construct_img(self):
        """_summary_"""
        dimentions = [(width, height) for width, height, row in self.matrix.imgsrows]
        rows_widths, rows_heights = zip(*dimentions)
        edges_buffer = self.pixelspace * 2
        sprite_width = max(rows_widths) + edges_buffer
        sprite_height = sum(rows_heights) + edges_buffer
        self.img = Image.new("RGBA", (sprite_width, sprite_height), (255, 255, 255, 1))

    def populate_img(self):
        """_summary_"""
        imgs_coords = {}
        horizontal_space = self.pixelspace
        for row_width, row_height, row in self.matrix.imgsrows:
            horizon_alighn = round((self.img.width - row_width) / 2)
            img_left_x = horizon_alighn
            row_coords = self.populate_row(row, img_left_x, row_height, horizontal_space)
            imgs_coords.update(row_coords)
            self.pixelspace += row_height
        self.imgscoords = imgs_coords

    def populate_row(self, row_imgs: list[Img], left_x: float, height: float, horizontal_space) -> dict:
        """_summary_"""
        row_imgs_coords = {}
        for img in row_imgs:
            vertical_alighn = round((height - img.height) / 2)
            img_upper_y = self.pixelspace + vertical_alighn
            box = left_x, img_upper_y
            # box = left_x, img_upper_y, left_x + img.width, img_upper_y - img.height
            self.img.paste(img.img, box)
            left_x += img.width + horizontal_space
            row_imgs_coords[img.name] = (left_x, img_upper_y, img.width, img.height) # type: ignore
        return row_imgs_coords

    def generate_lowerimg(self):
        """_summary_"""
        multiple_dimentions = (int(dim * self.lowerfactor) for dim in self.img.size)
        multiple_img = self.img.resize(multiple_dimentions)
        self.lowerimg = multiple_img

    def save(self, output_dir):
        """_summary_"""
        img_path = join(output_dir, "sprite")
        self.img.save(f"{img_path}.png")
        lowerimg_path = f"{img_path}@x{self.lowerfactor}.png"
        self.lowerimg.save(lowerimg_path)


@dataclass
class SpriteJSON:
    """_summary_"""

    spriteimg: SpriteImage
    jsondict: dict = field(init=False)
    lowerfactor = 2
    lowerjsondict: dict = field(init=False)

    def __post_init__(self):
        self.generate_json()
        self.generate_lowerjsondict()

    def generate_json(self):
        """_summary_"""
        sprite_json = {}
        for name, (left_x, upper_y, width, height) in self.spriteimg.imgscoords.items():
            img_dict = {}
            img_dict["width"] = width
            img_dict["height"] = height
            img_dict["x"] = left_x
            bottom_y = self.upper_to_bottom_y(upper_y)
            img_dict["y"] = bottom_y
            img_dict["pixelRatio"] = 1
            sprite_json[name] = img_dict
        self.jsondict = sprite_json

    def upper_to_bottom_y(self, upper_y: float) -> O[float]:
        """_summary_"""
        if self.spriteimg.matrix.shape:
            matrix_height = self.spriteimg.matrix.shape[1]
            bottom_y = matrix_height * 2 - upper_y
            return bottom_y
        return None

    def generate_lowerjsondict(self):
        """_summary_"""
        lowerjsondict = self.jsondict.copy()
        keys_for_update = "width", "height", "x", "y"
        keys = [(key, subkey) for key in lowerjsondict for subkey in keys_for_update]
        for key, subkey in keys:
            lowerjsondict[key][subkey] /= self.spriteimg.lowerfactor
        self.lowerjsondict = lowerjsondict

    def save(self, output_dir):
        """_summary_"""
        json_path = join(output_dir, "sprite")
        with open(f"{json_path}.json", "w", encoding="utf8") as output:
            dump(self.jsondict, output)
        lower_json_path = f"{json_path}@x{self.lowerfactor}"
        with open(f"{lower_json_path}.json", "w", encoding="utf8") as output_lower:
            dump(self.lowerjsondict, output_lower)


class SpriteGenerator:
    """_summary_"""

    def __init__(self, rules: list, output_dir: str):
        """_summary_"""
        self.rules: list = rules
        self.output_dir: str = output_dir
        self.lower_factor: int = 2
    
    def generate(self) -> O[str]:
        """---- Class Main Method ----"""
        self.add_rules_properties()
        imgs = self.get_symbol_imgs()
        if not imgs:
            return None
        matrix = SpriteMatrix(imgs)
        sprite_img = SpriteImage(matrix)
        sprite_json = SpriteJSON(sprite_img)
        self.save_files(sprite_img, sprite_json)
        return self.output_dir

    def add_rules_properties(self):
        """_summary_"""
        for rule in self.rules:
            if isinstance(rule, QgsRuleBasedRenderer.Rule):
                rule.type = 0
                rule.rulesymbol = rule.symbol()
            else:
                rule.type = 1
                rule.rulesymbol = rule.settings().format().background().markerSymbol()


    def get_symbol_imgs(self) -> list[SymbolImage]:
        """_summary_"""
        imgs = []
        for rule in self.rules:
            if self.rule_lyr_required_sprites(rule):
                symbol_img = SymbolImage(rule.rulesymbol, rule.description())
                imgs.append(symbol_img)
        return imgs

    @staticmethod
    def rule_lyr_required_sprites(rule) -> bool:
        """_summary_"""
        if not hasattr(rule, 'rulesymbol'):
            return False
        if rule.type == 1:
            return True
        symbol_lyr = rule.rulesymbol.symbolLayers()[0]
        if symbol_lyr.type() == QgsSymbol.SymbolType.Marker:
            return True
        subymbol = symbol_lyr.subSymbol()
        if subymbol and subymbol.type() == QgsSymbol.SymbolType.Marker:
            return True
        return False

    def save_files(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """_summary_"""
        sprite_img.save(self.output_dir)
        sprite_json.save(self.output_dir)


if __name__ == "__console__":

    output_dir = r'C:\Users\P0026701\OneDrive - Ness Israel\Desktop\ScratchWorkspace'
    rules = []

    def fetch_rules(rule):
        rules.append(rule)
        if rule.children():
            for child in rule.children():
                fetch_rules(child)
    layer = iface.activeLayer()
    for child in layer.renderer().rootRule().children() + layer.labeling().rootRule().children():
        fetch_rules(child)
    generator = SpriteGenerator(rules, output_dir)
    generator.generate()

    