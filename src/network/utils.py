import matplotlib.pyplot as plt
import numpy as np


def gerar_grafico(x: list[float], y: list[float], xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    # Gerando array numpy
    x = np.array(x)
    y = np.array(y)
    # Criando plot
    plt.plot(x, y)
    # Configurando labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Mostrando gráfico
    plt.show()
