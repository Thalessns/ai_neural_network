from fastapi import FastAPI
from src.network.service import NeuralNetwork

app = FastAPI(
    title="Trabalho de Inteligência Artificial - Rede Neural LMP",
    version="0.1.0"
)


@app.get("/")
async def root():
    return "{'Hello':'Service is alive and running!'}"


@app.post("/test")
async def test():
    network = NeuralNetwork()
    result = await network.test()
    await network.iniciar()
