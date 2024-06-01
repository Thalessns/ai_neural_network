from pydantic import BaseModel
from typing import Dict, List


class NetworkInput(BaseModel):
    performance_evaluation: str
    input_size: int
    hidden_size: int
    output_size: int
    dropout_rate: float
    lambda_reg: float
    max_epochs: int
    initial_learning_rate: float
    learning_rate_function: str
    activation_functions: Dict[str, str]
    imgs_source: str
    label_source: str


class NetworkEpochInput(BaseModel):
    epoca_id: int
    imgs_source: str
    label_source: str


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
