from pydantic import BaseModel
from typing import Dict, List


class NetworkInput(BaseModel):
    input_size: int
    hidden_size: int
    output_size: int
    initial_learning_rate: float
    learning_rate_function: str
    activation_functions: Dict[str, str]
    

class TreinamentoInput(BaseModel):
    input_size:            int
    hidden_size:           int
    hidden_weights:        List[List[float]]
    output_weights:        List[List[float]]
    output_size:           int
    initial_learning_rate: float
    activation_functions:  List[str]
    learning_rate_function: str
    accuracy: float

class Treinamento(TreinamentoInput):
    epoca: int
