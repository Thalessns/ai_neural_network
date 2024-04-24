from typing import Union, Optional, List
from src.neuron.functions import functions


class EntryNeuron():

    def __init__(
        self, 
        layer:  int,
        number: int,
        input:  float,
        output: float = None
        ) -> None:
        # Definindo variáveis iniciais
        self.layer  = layer
        self.number = number
        self.input  = input
        self.output = self.input if output is None else output

    async def activate(self) -> float:
        pass
    

class HiddenNeuron(EntryNeuron):

    def __init__(
        self, 
        layer:  int,
        number: int,
        weight: float = None,
        bias:   float = 0,
        input:  Union[float, None] = None,
        output: Union[float, None] = None
        ) -> None:
        # Definindo variáveis iniciais
        super().__init__(layer=layer, number=number, input=input, output=output)
        self.weight = weight
        self.bias   = bias
    
    async def activate(self) -> float:
        self.output = functions.sigmoid(x=self.input)
        return self.output

class ExitNeuron(HiddenNeuron):

    def __init__(
        self, 
        layer:  int,
        number: int,
        weight: Union[float, None] = None,
        bias:   float = 0,
        input:  Union[float, None] = None,
        output: Union[float, None] = None
        ) -> None:
        # Definindo variáveis iniciais
        super().__init__(layer=layer, number=number)
        
    async def activate(self) -> float:
        return self.input * self.weight


class NeuronFunctions():

    async def create_entry_neurons(self, inputs: list) -> List[EntryNeuron]: # Criando neurônios da camada de entrada
        return [EntryNeuron(layer=1, number=i+1, input=value) for i, value in enumerate(inputs)]

    async def create_hidden_neurons(self, layer: int, weights: list, n: int) -> List[HiddenNeuron]: # Criando neurônios da camada escondida
        return [HiddenNeuron(layer=layer, number=i+1, weight=weights[i]) for i in range(0, n)]
    
    async def create_exit_neurons(self,  layer: int, n: int) -> List[ExitNeuron]: # Criando neurônios da camada de saída
        return [ExitNeuron(layer=layer, number=i+1) for i in range (0, n)]
    
neuron_functions = NeuronFunctions()