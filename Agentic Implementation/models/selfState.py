from dataclasses import dataclass
from models.stateType import StateType

@dataclass
class SelfState:
    text: str
    state_type: StateType
    confidence: float
    reasoning: str
    span_start: int
    span_end: int