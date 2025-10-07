from abc import ABC, abstractmethod
from models


class SelfStateClassifier(ABC):
    """Abstract base class for self-state classifiers"""
    
    @abstractmethod
    def predict(self, post: RedditPost) -> List[SelfStateEvidence]:
        """Predict self-state evidence in a post"""
        pass