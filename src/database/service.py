from sqlalchemy import select, insert
from json import dumps, loads
from typing import Union, Dict, List

from src.database.tables import treinamento
from src.database.utils import execute, fetch_all, fetch_one
from src.network.schemas import TreinamentoInput, Treinamento


class Database:
    
    async def insert(self, data: TreinamentoInput) -> None:
        insert_query = insert(treinamento).values(
            input_size             = data.input_size,
            hidden_size            = data.hidden_size,
            hidden_weights         = dumps(data.hidden_weights),
            output_size            = data.output_size,
            output_weights         = dumps(data.output_weights),
            initial_learning_rate  = data.initial_learning_rate,
            activation_functions   = dumps(data.activation_functions),
            learning_rate_function = data.learning_rate_function,
            accuracy               = data.accuracy
        )
        await execute(insert_query)

    async def fix_rows(self, result: List[Dict]) -> Union[List[Dict], Dict]:
        rows = list()
        for i, row in enumerate(result): 
            rows.append({key: value for key, value in row.items()})
            rows[i]["hidden_weights"] = loads(rows[i]["hidden_weights"]) 
            rows[i]["output_weights"] = loads(rows[i]["output_weights"]) 
            rows[i]["activation_functions"] = loads(rows[i]["activation_functions"])
        return rows
    
    async def select_all(self) -> List[Treinamento]:
        select_query = select(treinamento)
        result = await fetch_all(select_query)
        rows = await self.fix_rows(result)
        return [Treinamento(**row) for row in rows] 
    
    async def select(self, epoca: int) -> Treinamento:
        select_query = select(treinamento).where(treinamento.epoca == epoca)
        result = await fetch_all(select_query)
        row = await self.fix_rows(result)
        return Treinamento(**row[0])


database = Database()