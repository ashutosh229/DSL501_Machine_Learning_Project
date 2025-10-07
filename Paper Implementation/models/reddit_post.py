from dataclasses import dataclass, field  
from typing import List  

@dataclass
class Post:
    """Represents a Reddit post with annotations"""
    post_id: str
    text: str
    adaptive_evidence: List[str] = field(default_factory=list)
    maladaptive_evidence: List[str] = field(default_factory=list)
    summary: str = ""
    wellbeing_score: float = 0.0