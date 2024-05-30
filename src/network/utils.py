import matplotlib.pyplot as plt
import numpy as np
from typing import List
from src.loader.service import Loader


def gerar_grafico(x: list[float], y: list[float], xlabel: str = "", ylabel: str = "", title: str = "") -> plt:
    # Gerando array numpy
    x = np.array(x)
    y = np.array(y)
    # Criando plot
    plt.plot(x, y)
    # Configurando labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def separa_dados_por_letras(data: List[List[int]], labels: List[List[int]]) -> dict[str, List[List[int]]]:
    """Retorna um dicionário com letras como chave e os dados correspondentes como valor"""
    result = {}
    for i in range(len(labels)):
        key = Loader.converter_binario_rotulo(labels[i])
        if key not in result:
            result[key] = []
        result[key].append(data[i])

    return result


async def distribui_valores(data: List[List[int]], labels: List[List[int]], percentage_train, percentage_validation):
    """Divide os dados em treino, validação e teste com a mesma distribuição de letras"""
    letras_dados = separa_dados_por_letras(data, labels)
    train_array = []
    train_labels = []
    validation_array = []
    validation_labels = []
    test_array = []
    test_labels = []

    for letter, values in letras_dados.items():
        total_count = len(letras_dados[letter])
        train_count = round(total_count * percentage_train)
        validation_count = round(total_count * percentage_validation)
        test_count = total_count - train_count - validation_count

        letra_em_binario = await Loader.converter_rotulos([letter])

        train_array.extend(values[:train_count])
        train_labels.extend(letra_em_binario * train_count)
        validation_array.extend(values[train_count:train_count + validation_count])
        validation_labels.extend(letra_em_binario * validation_count)
        test_array.extend(values[train_count + validation_count:])
        test_labels.extend(letra_em_binario * test_count)

    return train_array, train_labels, validation_array, validation_labels, test_array, test_labels
