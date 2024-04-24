import numpy
from src.neuron.classes import neuron_functions
from src.loader.service import loader

class NeuralNetwork():

    def __init__(self):
        pass

    async def iniciar(self):
        # Pegando dados para a rede
        data = await loader.carregar_imagem("src/files/X_png/0.png")
        
        # Definindo inputs
        inputs = [1, -1]

        # Neurônios de entrada
        entry_neurons = await neuron_functions.create_entry_neurons(inputs=inputs)



        ultimos_outputs = []

        weights = [0.3, 0.5]
        bias = 0
        fim = 2
        # Neurônios da camada escondida
        
        exit_neurons = await neuron_functions.create_exit_neurons(
            layer = fim,
            n     = 1
        )
        
        sum = 0
        for i in range(0, len(inputs)):
            print(sum)
            sum += inputs[i] * weights[i]

        result = bias + sum

        print(result)

        return 1 if result >= 0 else 0
    
    async def test(self):
        inputs = [1,-1]
                #-1,1,-1
                #1,-1,-1
                #1,1,1]
        
        # Neurônios de entrada
        entry_neurons = await neuron_functions.create_entry_neurons(inputs=inputs)

        for neuron in entry_neurons:
            print(f"""
                layer: {neuron.layer} {type(neuron.layer)}
                number: {neuron.number} {type(neuron.number)}
                input: {neuron.input} {type(neuron.input)}
                output: {neuron.output} {type(neuron.output)}
            """)