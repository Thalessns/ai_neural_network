from src.neuron.schemas import (EntryNeuron, HiddenNeuron, ExitNeuron)
from typing import List

class NeuronFunctions():

    async def create_entry_neurons(inputs: list) -> List[EntryNeuron]: # Criando neurônios da camada de entrada
        return [EntryNeuron(layer=1, number=i, input=value) for i, value in enumerate(inputs)]
    
    async def create_hidden_neurons(layer: int, weight: list) -> List[HiddenNeuron]: # Criando neurônios da camada escondida
        return [HiddenNeuron(layer=layer, number=i, weight=value) for i, value in enumerate(weight)]
    
    async def create_exit_neurons(layer: int, weight: list) -> List[ExitNeuron]: # Criando neurônios da camada de saída
        return [ExitNeuron(layer=layer, number=i, weight=value) for i, value in enumerate(weight)]
    

neuron_class = NeuronFunctions()