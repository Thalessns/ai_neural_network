from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio.engine import create_async_engine
from typing import Union, List, Dict
from src.database.config import db_config

Base = declarative_base(metadata=MetaData())
engine = create_async_engine(url=db_config.conn_url)


async def fetch_all(query) -> Union[List[Dict], None]:
    cursor = await execute(query)
    rows = cursor.fetchall()
    if len(rows) < 1:
        return Exception("No result found")
    return [row._mapping for row in rows]


async def fetch_one(query) -> Dict:
    cursor = await execute(query)
    res = cursor.fetchone()
    if res is None:
        return Exception("No result found")
    return res


async def execute(query) -> None: 
    async with engine.begin() as conn:
        await conn.execute(query)


async def init_models() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        