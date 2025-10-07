from dataclasses import dataclass

@dataclass
class SelfStateSpan:
    """Represents a self-state evidence span"""
    text: str
    label: str  # 'adaptive' or 'maladaptive'
    start_idx: int = -1
    end_idx: int = -1
    confidence: float = 1.0