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
                                        activation_function=functions.sigmoid,
                                        len_input=hidden_size)

        # TODO: Implementar feed_forward e back_propagation através das camadas

    async def iniciar(self):
        # Pegando dados para a rede
        data = await loader.carregar_imagem("src/files/X_png/0.png")

        # Transformando em uma lista de valores
        data = [item for sublist in data for item in sublist]

        # Definindo inputs
        inputs = [-1, 1]
        return 1

    async def test(self):
        pass
