import random
from typing import Callable
from src.neuron.functions import functions


class Layer:
    def __init__(self, num_neurons: int, activation_function: Callable) -> None:
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.weights = []
        self.biases = []

    def init_weights(self, num_inputs) -> None:
        """Inicializa o vetor de pesos com valores aleatórios entre -0.5 e 0.5 e o vetor de biases com 0"""
        for _ in range(self.num_neurons):
            self.weights.append([random.uniform(-0.5, 0.5) for _ in range(num_inputs)])
            self.biases.append(0)

    async def feed_forward(self, inputs: list) -> list:
        outputs = []
        for neuron_weight, bias in zip(self.weights, self.biases):
            soma_ponderada = sum(weight * input for weight, input in zip(neuron_weight, inputs)) + bias
            outputs.append(await self.activation_function(soma_ponderada))
        return outputs


# testes
async def testes_layer():
    inputs = [1.0, 2.0, 3.0]
    camada = Layer(num_neurons=2, activation_function=functions.relu)
    camada.init_weights(num_inputs=len(inputs))
    i = 0
    for pesos in camada.weights:
        print(f"neuronio {i} com pesos:{pesos}")
        i += 1
    print("-----------")
    print(await camada.feed_forward(inputs))
