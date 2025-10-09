from dataclasses import dataclass  
from typing import List, Dict, Any  
from models.selfstate import SelfState

@dataclass
class ClassificationResult:
    post_id: str
    self_states: List[SelfState]
    overall_confidence: float
    processing_time: float
    agent_votes: Dict[str, Any]