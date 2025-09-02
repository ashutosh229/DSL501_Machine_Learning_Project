from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class StateType(Enum):
    ADAPTIVE = "adaptive"
    MALADAPTIVE = "maladaptive"
    NEUTRAL = "neutral"


@dataclass
class SelfState:
    text: str
    state_type: StateType
    confidence: float
    reasoning: str
    span_start: int
    span_end: int


@dataclass
class ProcessedPost:
    original_text: str
    sentences: List[str]
    context_windows: List[str]
    user_id: str
    post_id: str
    timeline_position: int


@dataclass
class ClassificationResult:
    post_id: str
    self_states: List[SelfState]
    overall_confidence: float
    processing_time: float
    agent_votes: Dict[str, Any]
