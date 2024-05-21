from fastapi import APIRouter
from typing import Any

from src.network.service import NeuralNetwork
from src.network.schemas import NetworkInput, NetworkEpochInput
from src.neuron.utils import utils
from src.database.service import database

network_router = APIRouter(prefix="/network")


@network_router.post("/treinar", response_model=Any)
async def treinar(input: NetworkInput) -> Any:
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
        learning_rate_function=learning_rate_function
    )
    return await rede.treinar(input.imgs_source, input.label_source)


@network_router.post("/treinar-epoca", response_description=Any)
async def treinar_epoca(input: NetworkEpochInput) -> Any:
    # Obtendo objetos das funções
    epoca = await database.select(input.epoca_id)
    epoca = epoca.model_dump()
    # Obtendo objetos das funções
    epoca["activation_functions"] = utils.verify_activation_functions(epoca["activation_functions"])
    epoca["learning_rate_function"] = utils.verify_learning_rate_function(epoca["learning_rate_function"])
    # Removendo atributos desnecessários
    del epoca["epoca"]
    del epoca["accuracy"]
    # Criando instância da rede neural
    rede = NeuralNetwork(**epoca)
    # Executando treinamento
    return await rede.treinar(input.imgs_source, input.label_source)