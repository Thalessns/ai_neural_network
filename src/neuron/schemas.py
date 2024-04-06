from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class EntryNeuron():
    layer:  int
    number: int
    input:  Optional[Union[float, None]] = field(default=None)
    output: Optional[Union[int, None]] = field(default=None)


@dataclass
class HiddenNeuron(EntryNeuron):
    weight: float


@dataclass
class ExitNeuron(HiddenNeuron):
    pass