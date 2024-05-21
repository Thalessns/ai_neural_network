from fastapi import FastAPI
from typing import Dict, Any

from src.database.utils import init_models
from src.database.router import database_router
from src.network.router import network_router

app = FastAPI(
    title="Trabalho de Inteligência Artificial - Rede Neural LMP",
    version="0.1.0"
)


@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    return {'Hello':'Service is alive and running!'}

# Iniciando banco de dados
app.add_event_handler("startup", init_models)
# Adicionando rotas da rede neural
app.include_router(database_router)
app.include_router(network_router)
