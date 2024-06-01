from fastapi import APIRouter
from typing import Any

from src.network.service import NeuralNetwork
from src.network.schemas import NetworkInput
from src.neuron.utils import utils

network_router = APIRouter(prefix="/network")


@network_router.post("/treinar", response_model=Any)
async def test(input: NetworkInput) -> Any:
    # Obtendo objetos das funções
    activation_functions = utils.verify_activation_functions(input.activation_functions.values())
    learning_rate_function = utils.verify_learning_rate_function(input.learning_rate_function)
    # Criando instância da rede neural
    rede = NeuralNetwork(
        input_size=input.input_size, 
        hidden_size=input.hidden_size, 
        output_size=input.output_size, 
        activation_functions=activation_functions,
        initial_learning_rate=input.initial_learning_rate,
        learning_rate_function=learning_rate_function,
        dropout_rate=input.dropout_rate,
        max_epochs=input.max_epochs
    )
    if input.performance_evaluation == "holdout":
        return await rede.train_holdout(input.imgs_source, input.label_source)
    if input.performance_evaluation == "cross_validation":
        return await rede.train_cross_validation(input.imgs_source, input.label_source)
    return {"error": "Model evaluation not found"}
