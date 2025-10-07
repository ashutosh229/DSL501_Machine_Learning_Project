from dataclasses import dataclass, field  
from typing import List  
from models.redditPost import Post

@dataclass
class Timeline:
    """Represents a user's timeline of posts"""
    user_id: str
    posts: List[Post] = field(default_factory=list)
    summary: str = ""