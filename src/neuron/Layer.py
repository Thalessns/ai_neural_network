import random
from typing import Callable
from src.neuron.functions import functions


class Layer:
    def __init__(self, num_neurons: int, activation_function: Callable) -> None:
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.weights = []
        self.bias = []

    def init_weights(self, num_inputs) -> None:
        for _ in range(self.num_neurons):
            self.weights.append([random.uniform(-0.5, 0.5) for i in range(num_inputs)])
            self.bias.append(0)


# testes
if __name__ == "__main__":
    camada = Layer(num_neurons=3, activation_function=functions.relu())
    camada.init_weights(num_inputs=5)
    print(camada.weights)
    print(camada.bias)