from sqlalchemy import (Column, Integer, Float, Text)
from src.database.utils import Base


class Treinamentos(Base):
    __tablename__ = "treinamentos"

    epoca           = Column(Integer, primary_key=True)
    neuronios       = Column(Text, nullable=False)
    modificados_qtd = Column(Text, nullable=False)
    erro_medio      = Column(Float, nullable=False)
    source          = Column(Text, nullable=False)