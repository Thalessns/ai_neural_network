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
        key = Loader.converter_binario_para_letra(labels[i])
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


def imprimir_matriz_confusao(lista_true: list[str], lista_output: list[str]):
    # Inicializar a matriz de confusão 26x26
    confusion_matrix = [[0 for _ in range(26)] for _ in range(26)]

    # Criar um dicionário para mapear letras minúsculas para índices
    letra_para_indice = {chr(i): i - 97 for i in range(97, 123)}  # a-z -> 0-25

    # Preencher a matriz de confusão
    for true, pred in zip(lista_true, lista_output):
        true_index = letra_para_indice[true]
        pred_index = letra_para_indice[pred]
        confusion_matrix[true_index][pred_index] += 1

    # Imprimir a matriz de confusão
    print("Matriz de Confusão:")
    print("     " + " ".join([chr(i) for i in range(97, 123)]))  # Header
    for i in range(26):
        row = " ".join([str(confusion_matrix[i][j]) for j in range(26)])
        print(f"{chr(i + 97)}    {row}")
