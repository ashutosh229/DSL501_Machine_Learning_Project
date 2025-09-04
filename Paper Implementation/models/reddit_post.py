from dataclasses import dataclass
from typing import List

@dataclass
class RedditPost:
    post_id: str
    text: str
    adaptive_evidence: List[str] = None
    maladaptive_evidence: List[str] = None
    summary: str = ""
    well_being_score: float = 0.0