from src.loader.service import loader
from src.neuron.layer import InputLayer, HiddenLayer, OutputLayer
from src.neuron.functions import functions


class NeuralNetwork:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, initial_learning_rate: float):
        self.learning_rate = initial_learning_rate
        self.input_layer = InputLayer()
        self.hidden_layer = HiddenLayer(num_neurons=hidden_size,
                                        activation_function=functions.tanh,
                                        len_input=input_size)
        self.output_layer = OutputLayer(num_neurons=output_size,
                                        activation_function=functions.tanh,
                                        len_input=hidden_size)

    @staticmethod
    async def compute_mean_squared_error(all_outputs: list[list[float]], all_expected_outputs: list[list[float]]):
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

    async def get_output(self, entrada: list[float]) -> list[float]:
        """Dado uma entrada, calcula a saída da rede neural"""

        hidden_inputs = await self.input_layer.feed_forward(entrada)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        output = await self.output_layer.feed_forward(hidden_outputs)
        return output

    async def train_one_sample(self, inputs: list[float], expected_outputs: list[float]):
        """Treina a rede neural com um exemplo de entrada e saída esperada"""

        # Feed forward
        hidden_inputs = await self.input_layer.feed_forward(inputs)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        outputs = await self.output_layer.feed_forward(hidden_outputs)

        # Back propagation
        hidden_error = await self.output_layer.back_propagation(inputs=hidden_outputs,
                                                                outputs=outputs,
                                                                expected_outputs=expected_outputs,
                                                                learning_rate=self.learning_rate)
        await self.hidden_layer.back_propagation(inputs=hidden_inputs,
                                                 erros_output=hidden_error,
                                                 learning_rate=self.learning_rate)

    async def do_one_epoch(self, inputs: list[list[float]], expected_outputs: list[list[float]]):
        """Treina uma época da rede neural"""

        for entrada, expected in zip(inputs, expected_outputs):
            await self.train_one_sample(inputs=entrada, expected_outputs=expected)

    async def iniciar(self):
        # Pegando dados para a rede
        data = await loader.carregar_imagem("src/files/X_png/0.png")

        # Transformando em uma lista de valores
        data = [item for sublist in data for item in sublist]

        # Definindo problema XOR
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected_outputs = [[0], [1], [1], [0]]

        # Treinando a rede
        for epoch in range(10000):
            await self.do_one_epoch(inputs=inputs, expected_outputs=expected_outputs)

            # imprime resultados
            if epoch % 100 == 0:
                resultados = []
                print(f"epoca {epoch}")
                for entrada in inputs:
                    resultados.append(await self.get_output(entrada=entrada))
                    print(f"entrada {entrada}, saida da rede {await self.get_output(entrada=entrada)}")
                mse = await self.compute_mean_squared_error(all_outputs=resultados,
                                                            all_expected_outputs=expected_outputs)
                print(f"mse: {mse}")
                print(f"camada oculta pesos: {self.hidden_layer.weights} - biases: {self.hidden_layer.biases}")
                print(f"camada saída pesos: {self.output_layer.weights} - biases: {self.output_layer.biases}")
                if mse < 0.001:
                    break

        return 1
