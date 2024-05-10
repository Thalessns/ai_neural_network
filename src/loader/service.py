from PIL import Image
from typing import List


class Loader():

    async def carregar_imagem(self, path: str) -> List[List]:
        # Abrundo imagem
        image = Image.open(path) 
        # Obtendo dimensões
        largura, altura = image.size 
        # Convertendo para escala de cinza
        image = image.convert("L") 
        # Obtendo pixels em uma lista de tuplas (x, y)
        pixels_raw = list(image.getdata()) 
        threshold = 128  # Valor de limiar para converter em binário
        pixels = [0 if pixel < threshold else 1 for pixel in pixels_raw]
        # Converter as listas de pixels em uma matriz 2D
        return [pixels[i * largura:(i + 1) * largura] for i in range(altura)]


loader = Loader()