import random
from typing import Callable
from src.neuron.functions import functions


class InputLayer:
    @staticmethod
    def feed_forward(data) -> list[float]:
        """Passa os dados de entrada para a próxima camada"""
        return data


class HiddenLayer:
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int) -> None:
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.derivative_function = functions.get_derivative_function(activation_function)
        self.weights: list[list[float]] = []
        self.biases: list[float] = []

        self.init_weights(len_input)

    def init_weights(self, len_input: int) -> None:
        """Inicializa o vetor de pesos com valores aleatórios entre -0.5 e 0.5 e o vetor de biases com 0"""
        for _ in range(self.num_neurons):
            self.weights.append([random.uniform(-0.5, 0.5) for _ in range(len_input)])
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


class OutputLayer(HiddenLayer):
    def __init__(self, num_neurons: int, activation_function: Callable, len_input: int):
        super().__init__(num_neurons, activation_function, len_input)

    # TODO: reescrever feed_forward e back_propagation para a camada de saída


# testes
async def testes_layer():
    # inputs = [random.randint(-10, 10) for _ in range(random.randint(10, 50))]
    # expected_output = [random.uniform(0, 1) for _ in range(10)]
    # AND inputs
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # AND outputs
    expected_outputs = [[-1], [-1], [-1], [1]]
    camada = HiddenLayer(num_neurons=1, activation_function=functions.tanh, len_input=len(inputs[0]))

    learning_rate = 0.01
    epochs = 25000

    last_average_mse = 0

    for epoch in range(epochs):
        for input, expected_output in zip(inputs, expected_outputs):
            output = await camada.feed_forward(input)
            await camada.back_propagation(input, output, expected_output, learning_rate)

        # printa informações a cada 50 épocas
        if epoch % 50 == 0:
            # Calcula a média do erro quadrático médio
            total_mse = 0
            for input, expected_output in zip(inputs, expected_outputs):
                output = await camada.feed_forward(input)
                mse = await camada.compute_mean_squared_error(output, expected_output)
                total_mse += mse
            avg_mse = total_mse / len(inputs)

            print(f"""
            Epoch: {epoch}
            Weights: {camada.weights}
            Biases: {camada.biases}
            MSE: {avg_mse}""")
            if abs(avg_mse - last_average_mse) < 0.00001:
                break
            last_average_mse = avg_mse

    # testes finais
    print("""
    ----------------------------
    Testes finais""")
    for input, expected_output in zip(inputs, expected_outputs):
        output = await camada.feed_forward(input)
        print(f"In: {input} Output: {output} Expected Output: {expected_output}")
