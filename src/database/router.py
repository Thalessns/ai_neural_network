from fastapi import APIRouter
from typing import List

from src.network.schemas import Treinamento, TreinamentoInput
from src.database.service import database

database_router = APIRouter(prefix="/database")


@database_router.get("/select-all", response_model=List[Treinamento])
async def test() -> List[Treinamento]:
    return await database.select_all()


@database_router.get("/select", response_model=Treinamento)
async def test(epoca: int) -> Treinamento:
    return await database.select(epoca)
