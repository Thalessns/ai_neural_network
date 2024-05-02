import numpy
from src.neuron.classes import neuron_functions
from src.loader.service import loader


class NeuralNetwork:

    def __init__(self, initial_learning_rate: int):
        self.learning_rate = initial_learning_rate

    async def iniciar(self):
        # Pegando dados para a rede
        data = await loader.carregar_imagem("src/files/X_png/0.png")

        # Definindo inputs
        inputs = [-1, 1]

        # Neurônios de entrada
        entry_neurons = await neuron_functions.create_entry_neurons(inputs=inputs)

        ultimos_outputs = []

        weights = [0.3, 0.8]
        bias = 0
        ultima_camada = 2
        # Neurônios da camada escondida

        exit_neurons = await neuron_functions.create_exit_neurons(
            layer=ultima_camada,
            n=1
        )

        somatorio = sum(input_val * weight_val for input_val, weight_val in zip(inputs, weights))
        output = bias + somatorio

        return 1 if output >= 0 else 0

    async def test(self):
        #testar com o xor
        inputs = [1, -1]
        # -1,1,-1
        # 1,-1,-1
        # 1,1,1]

        # Neurônios de entrada
        entry_neurons = await neuron_functions.create_entry_neurons(inputs=inputs)

