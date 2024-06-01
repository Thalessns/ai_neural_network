from PIL import Image
from typing import List
import os


class Loader:

    @staticmethod
    async def carregar_imagem(path: str) -> List[int]:
        """Carregar imagem e converter para binário"""
        # Abrindo imagem
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
        data = [pixels[i * largura:(i + 1) * largura] for i in range(altura)]
        # Transformando em uma lista de valores
        data = [item for sublist in data for item in sublist]
        return data

    @staticmethod
    async def carregar_todas_imagens(folder_path) -> List[List[int]]:
        """Carregar todas as imagens de uma pasta"""
        data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                data.append(await Loader.carregar_imagem(os.path.join(folder_path, filename)))
        return data

    @staticmethod
    async def carregar_todos_rotulos(path: str) -> List[List[int]]:
        """Carregar todos os rotulos e converter para binário"""
        labels = []
        with open(path, "r") as file:
            for line in file:
                labels.append(line.strip())

        return await Loader.converter_letras_para_binario(labels)

    @staticmethod
    async def converter_letras_para_binario(labels: List[str]) -> List[List[int]]:
        """Converte letras para binário"""
        resultado = []
        for letter in labels:
            letter = letter.lower()
            binario = [0] * 26
            binario[ord(letter) - ord("a")] = 1
            resultado.append(binario)

        return resultado

    @staticmethod
    def converter_binario_para_letra(binario: List[int]) -> str:
        """Converte binário para letra"""
        return chr(binario.index(1) + ord("a"))
