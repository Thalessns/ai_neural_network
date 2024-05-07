import random
from typing import Callable
from src.neuron.functions import functions


class Layer:
    def __init__(self, num_neurons: int, activation_function: Callable, num_inputs: int) -> None:
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.derivative_function = functions.get_derivative_function(activation_function)
        self.weights: list[list[float]] = []
        self.biases: list[float] = []

        self.init_weights(num_inputs)

    def init_weights(self, num_inputs: int) -> None:
        """Inicializa o vetor de pesos com valores aleatórios entre -0.5 e 0.5 e o vetor de biases com 0"""
        for _ in range(self.num_neurons):
            self.weights.append([random.uniform(-0.5, 0.5) for _ in range(num_inputs)])
            self.biases.append(0)

    @staticmethod
    async def compute_mean_squared_error(output: list[float], expected_output: list[float]):
        """Calcula o erro quadrático médio"""
        if len(output) != len(expected_output):
            raise ValueError("Listas de output com tamanhos diferentes")

        mean_squared_error = sum((value - expected_value) ** 2
                                 for value, expected_value in zip(output, expected_output))
        mean_squared_error /= len(output)
        return mean_squared_error

    async def feed_forward(self, inputs: list[float]) -> list[float]:
        """Realiza a operação de feed forward e retorna a saída da camada"""

        if len(self.weights) != len(self.biases):
            raise ValueError("Biases e pesos tamanhos diferentes")

        outputs = []
        for neuron_weight, bias in zip(self.weights, self.biases):
            soma_ponderada = sum(weight * input for weight, input in zip(neuron_weight, inputs)) + bias
            outputs.append(await self.activation_function(soma_ponderada))
        return outputs

    async def back_propagation(self,
                               inputs: list[float],
                               outputs: list[float],
                               expected_outputs: list[float],
                               learning_rate: float):
        """Realiza o backpropagation para atualizar os pesos e os biases da camada e retorna o erro da camada
        anterior"""

        if len(self.weights) != len(self.biases):
            raise ValueError("Biases e pesos tamanhos diferentes")
        if len(outputs) != len(expected_outputs):
            raise ValueError("Listas de saída com tamanhos diferentes")

        # Calculando o erro
        erros = [expected_output - output for expected_output, output in zip(expected_outputs, outputs)]
        # Calculando o gradiente
        informacao_erro = [await self.derivative_function(output) * error for output, error in zip(outputs, erros)]

        # Calculando a correção dos pesos para cada unidade de saída
        variacao_pesos = []
        variacao_bias = []
        for i in range(len(self.weights)):
            variacao_pesos.append([learning_rate * informacao_erro[i] * input for input in inputs])
            variacao_bias.append(learning_rate * informacao_erro[i])

        # Calculando o erro da camada anterior
        erros_camada_anterior = []
        for i in range(len(self.weights)):
            soma = sum(informacao_erro[i] * weight for weight in self.weights[i])
            erros_camada_anterior.append(soma)

        # Atualizando os pesos e os biases
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += variacao_pesos[i][j]
            self.biases[i] += variacao_bias[i]

        return erros_camada_anterior


# testes
async def testes_layer():
    inputs = [3, 8, 2, 6, 5]
    camada = Layer(num_neurons=2, activation_function=functions.sigmoid, num_inputs=len(inputs))
    expected_output = [0.24567, 0.83124]
    learning_rate = 0.01
    epochs = 2500
    erro_anterior = 0.0

    for _ in range(epochs):
        output = await camada.feed_forward(inputs)
        erro = await camada.compute_mean_squared_error(output, expected_output)
        pesos = [[f'{weight:.2f}' for weight in neuron] for neuron in camada.weights]
        if _ % 200 == 0:
            print(f"epoca: {_} - pesos: {pesos} - output: {output} - erro: {erro}")
        if erro == erro_anterior: # Se o erro não mudar, para o treinamento
            print(f"epoca: {_} - pesos: {pesos} - output: {output} - erro: {erro}")
            break
        erro_anterior = erro

        await camada.back_propagation(inputs, output, expected_output, learning_rate)

