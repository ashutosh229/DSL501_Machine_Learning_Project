from dataclasses import dataclass

@dataclass
class SelfStateEvidence:
    text: str
    label: str 
    start_pos: int = -1
    end_pos: int = -1
    confidence: float = 0.0

