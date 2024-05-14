import random
from typing import Callable
from src.neuron.functions import functions


class Camada:
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int) -> None:
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.derivative_function = functions.get_derivative_function(activation_function)
        self.weights: list[list[float]] = []
        self.biases: list[float] = []
        self.output_pre_ativacao: list[float] = []

        self.init_weights(len_input)

    def init_weights(self, len_input: int) -> None:
        """Inicializa o vetor de pesos com valores aleatórios entre -0.5 e 0.5 e o vetor de biases com 0"""
        for _ in range(self.num_neurons):
            self.weights.append([random.uniform(-0.5, 0.5) for _ in range(len_input)])
            self.biases.append(0)

    async def feed_forward(self, inputs: list[float]) -> list[float]:
        """Realiza a operação de feed forward e retorna a saída da camada"""

        if len(self.weights) != len(self.biases):
            raise ValueError("Biases e pesos tamanhos diferentes")

        outputs = []
        self.output_pre_ativacao = []
        for neuron_weight, bias in zip(self.weights, self.biases):
            soma_ponderada = sum(weight * yi for weight, yi in zip(neuron_weight, inputs)) + bias
            self.output_pre_ativacao.append(soma_ponderada)
            outputs.append(await self.activation_function(soma_ponderada))
        return outputs


class InputLayer:
    @staticmethod
    async def feed_forward(data) -> list[float]:
        """Passa os dados de entrada para a próxima camada"""
        return data


class HiddenLayer(Camada):
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int) -> None:
        super().__init__(num_neurons, activation_function, len_input)

    async def feed_forward(self, inputs: list[float]) -> list[float]:
        return await super().feed_forward(inputs)

    async def back_propagation(self, inputs: list[float], erros_output: list[float], learning_rate: float) -> None:
        """Realiza o backpropagation para atualizar os pesos e os biases da camada baseado nos erros da camada de
        saída"""

        if len(self.weights) != len(self.biases):
            raise ValueError("Biases e pesos tamanhos diferentes")

        derivadas = [await self.derivative_function(vj) for vj in self.output_pre_ativacao]

        gradientes = [erro_neuronio * derivada for erro_neuronio, derivada in zip(erros_output, derivadas)]

        delta_pesos = []
        delta_biases = []
        for gradiente in gradientes:
            delta_pesos.append([learning_rate * gradiente * yi for yi in inputs])
            delta_biases.append(learning_rate * gradiente)

        # atualização de pesos
        for neuron in range(len(self.weights)):
            for i in range(len(self.weights[neuron])):
                self.weights[neuron][i] += delta_pesos[neuron][i]
            self.biases[neuron] += delta_biases[neuron]


class OutputLayer(Camada):
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int):
        super().__init__(num_neurons, activation_function, len_input)

    async def feed_forward(self, inputs: list[float]) -> list[float]:
        """Realiza a operação de feed forward e retorna a saída da camada"""
        resultado = await super().feed_forward(inputs)
        return resultado

    async def back_propagation(self,
                               inputs: list[float],
                               outputs: list[float],
                               expected_outputs: list[float],
                               learning_rate: float) -> list[float]:
        """Realiza o backpropagation para atualizar os pesos e os biases da camada e retorna o erro da camada
        oculta"""

        if len(self.weights) != len(self.biases):
            raise ValueError("Biases e pesos tamanhos diferentes")
        if len(outputs) != len(expected_outputs):
            raise ValueError("Listas de saída com tamanhos diferentes")

        erros = [expected_output - output for expected_output, output in zip(expected_outputs, outputs)]

        if len(self.output_pre_ativacao) != len(erros):
            raise ValueError("erros e erros pre ativação com tamanhos diferentes")

        # Calculando derivadas da função de ativação sobre a soma ponderada
        derivadas = [await self.derivative_function(vj) for vj in self.output_pre_ativacao]

        # Calculando o gradiente para cada unidade de saída
        gradientes = [erro_neuronio * derivada for erro_neuronio, derivada in zip(erros, derivadas)]

        # Calculando a variação nos pesos
        delta_pesos = []
        delta_biases = []
        for gradiente in gradientes:
            delta_pesos.append([learning_rate * gradiente * yi for yi in inputs])
            delta_biases.append(learning_rate * gradiente)

        # Calculando o erro da camada anterior
        erros_camada_anterior = []
        for j in range(len(inputs)):
            erros_camada_anterior.append(sum(gradientes[k] * self.weights[k][j] for k in range(self.num_neurons)))

        # alteração de pesos
        for neuron in range(len(self.weights)):
            for i in range(len(self.weights[neuron])):
                self.weights[neuron][i] += delta_pesos[neuron][i]
            self.biases[neuron] += delta_biases[neuron]

        return erros_camada_anterior
