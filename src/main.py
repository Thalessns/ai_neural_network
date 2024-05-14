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
    rede = NeuralNetwork(input_size=2, hidden_size=5, output_size=1, initial_learning_rate=0.1)
    await rede.iniciar()
    return 100
