import random
from typing import Union, List, Callable

from src.neuron.utils import utils


class Camada:
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int,  weights: Union[List[List[float]], None]) -> None:
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.derivative_function = utils.get_derivative_function(activation_function)
        self.weights: List[List[float]] = []
        self.biases: list[float] = []
        self.output_pre_ativacao: list[float] = []

        if weights is None: self.init_weights(len_input)
        else: 
            self.biases = [0 for _ in range(self.num_neurons)]
            self.weights = weights

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

    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int,  weights: Union[List[List[float]], None]) -> None:
        super().__init__(num_neurons, activation_function, len_input, weights)

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
    
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int,  weights: Union[List[List[float]], None]) -> None:
        super().__init__(num_neurons, activation_function, len_input, weights)

    async def feed_forward(self, inputs: list[float]) -> list[float]:
        """Realiza a operação de feed forward e retorna a saída da camada"""
        resultado = await super().feed_forward(inputs)
        # binarizando a saída
        for i in range(len(resultado)):
            #print(resultado[i], end=" | ")
            if resultado[i] >= 0.8:
                resultado[i] = 1
            else:
                resultado[i] = -1
        return resultado

    async def back_propagation(
        self,
        inputs: list[float],
        outputs: list[float],
        expected_outputs: list[float],
        learning_rate: float
    ) -> list[float]:
        """Realiza o backpropagation para atualizar os pesos e os biases da camada e retorna o erro da camada
        oculta"""

        if len(self.weights) != len(self.biases):
            raise ValueError("Biases e pesos tamanhos diferentes")
        if len(outputs) != len(expected_outputs):
            raise ValueError("Listas de saída com tamanhos diferentes")

        erro = await self.compute_mean_squared_error([outputs], [expected_outputs])

        # Calculando derivadas da função de ativação sobre a soma ponderada
        derivadas = [await self.derivative_function(vj) for vj in self.output_pre_ativacao]

        # Calculando o gradiente para cada unidade de saída
        gradientes = [erro * derivada for derivada in derivadas]

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

    async def compute_mean_squared_error(
        self,
        all_outputs: list[list[float]],
        all_expected_outputs: list[list[float]]
    ) -> float:
        """Dado as saídas e saídas esperadas para o conjunto de dados completo, calcula o erro quadrático médio"""

        if len(all_outputs) != len(all_expected_outputs):
            raise ValueError("Listas de outputs com tamanhos diferentes")

        erros_neuronio = []
        for obtidos_neuronio, esperados_neuronio in zip(all_outputs, all_expected_outputs):
            erros_neuronio.append([obtido - esperado for obtido, esperado in zip(obtidos_neuronio, esperados_neuronio)])

        erro_instantaneo = []
        for erro_neuronioj in erros_neuronio:
            erro_instantaneo.append(sum(erro ** 2 for erro in erro_neuronioj) / 2.0)

        erro_quadratico_medio = sum(erro_instantaneo) / len(erro_instantaneo)

        return erro_quadratico_medio