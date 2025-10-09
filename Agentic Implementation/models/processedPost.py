from dataclasses import dataclass  
from typing import List

@dataclass
class ProcessedPost:
    original_text: str
    sentences: List[str]
    context_windows: List[str]
    user_id: str
    post_id: str
    timeline_position: int