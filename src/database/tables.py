from sqlalchemy import (Column, Integer, Float, JSON, Text)
from src.database.utils import Base


class Treinamento(Base):
    __tablename__ = "treinamento"

    epoca                  = Column(Integer, primary_key=True, autoincrement=True)
    input_size             = Column(Integer, nullable=False)
    hidden_size            = Column(Integer, nullable=False)
    hidden_weights         = Column(JSON, nullable=False)
    output_size            = Column(Integer, nullable=False)
    output_weights         = Column(JSON, nullable=False)
    initial_learning_rate  = Column(Float, nullable=False)
    activation_functions   = Column(JSON, nullable=False)
    learning_rate_function = Column(Text, nullable=False)
    accuracy               = Column(Float, nullable=False)


treinamento = Treinamento