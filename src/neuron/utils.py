import inspect
from typing import List, Callable

from src.neuron.functions import (
    ActivationFunctions, 
    LearningRateFunctions, 
    activation_functions
)


class Utils():

    ACTIVATION_FUNCIONS = inspect.getmembers(ActivationFunctions, predicate=inspect.isfunction)
    ACTIVATION_DICT = {nome: funcao for nome, funcao in ACTIVATION_FUNCIONS}

    LEARNING_FUNCTIONS = inspect.getmembers(LearningRateFunctions, predicate=inspect.isfunction)
    LEARNING_DICT = {nome: funcao for nome, funcao in LEARNING_FUNCTIONS}

    def verify_activation_functions(self, functions: list[str]) -> List[Callable]:
        callables = []
        for function in functions: 
            if function not in self.ACTIVATION_DICT.keys():
                raise Exception(f"Função de ativação {function} não existe!")
            callables.append(self.ACTIVATION_DICT[function])   
        return callables 
    
    def verify_learning_rate_function(self, function: str) -> Callable:
        if function not in self.LEARNING_DICT.keys():
            raise Exception(f"Função de learning rate {function} não existe!")
        return self.LEARNING_DICT[function]

    def get_derivative_function(self, function: Callable) -> Callable:
        map_function_to_derivative = {
            activation_functions.relu: activation_functions.derivative_relu,
            activation_functions.elu: activation_functions.derivative_elu,
            activation_functions.swish: activation_functions.derivative_swish,
            activation_functions.selu: activation_functions.derivative_selu,
            activation_functions.soft_plus: activation_functions.derivative_soft_plus,
            activation_functions.hard_swish: activation_functions.derivative_hard_swish,
            activation_functions.sigmoid: activation_functions.derivative_sigmoid,
            activation_functions.soft_sign: activation_functions.derivative_soft_sign,
            activation_functions.tanh: activation_functions.derivative_tanh,
            activation_functions.hard_sigmoid: activation_functions.derivative_hard_sigmoid,
            activation_functions.tanh_shrink: activation_functions.derivative_tanh_shrink,
            activation_functions.soft_shrink: activation_functions.derivative_soft_shrink,
            activation_functions.hard_shrink: activation_functions.derivative_hard_shrink
        }
        return map_function_to_derivative[function]


utils = Utils()