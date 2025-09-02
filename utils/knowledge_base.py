from models.states import StateType
from typing import List


class KnowledgeBase:
    def __init__(self):
        self.mind_framework = {
            "affect": {
                "adaptive": [
                    "calm",
                    "content",
                    "assertive",
                    "proud",
                    "justified_grief",
                ],
                "maladaptive": [
                    "anxious",
                    "hopeless",
                    "apathetic",
                    "aggressive",
                    "ashamed",
                    "depressed",
                ],
            },
            "behavior_others": {
                "adaptive": ["relational", "autonomous", "supportive"],
                "maladaptive": ["fight_flight", "controlling", "overcontrolled"],
            },
            "behavior_self": {
                "adaptive": ["self_care", "healthy_boundaries"],
                "maladaptive": ["self_neglect", "avoidance", "self_harm"],
            },
            "cognition_others": {
                "adaptive": ["supportive_perception", "related", "trusting"],
                "maladaptive": [
                    "detached_perception",
                    "overattached",
                    "autonomy_blocking",
                ],
            },
            "cognition_self": {
                "adaptive": ["self_compassion", "acceptance", "realistic_self_view"],
                "maladaptive": ["self_criticism", "unrealistic_expectations"],
            },
            "desire": {
                "adaptive": [
                    "autonomy_seeking",
                    "relatedness",
                    "self_esteem",
                    "care_seeking",
                ],
                "maladaptive": ["fear_unmet_needs", "desperation", "hopelessness"],
            },
        }

        self.adaptive_patterns = [
            "asking for help",
            "expressing gratitude",
            "setting boundaries",
            "self-reflection",
            "problem-solving",
            "seeking support",
            "making plans",
            "showing empathy",
        ]

        self.maladaptive_patterns = [
            "self-harm ideation",
            "isolation behaviors",
            "extreme self-criticism",
            "hopelessness",
            "suicidal thoughts",
            "destructive behaviors",
            "overwhelming emotions",
        ]

    def get_relevant_patterns(self, text: str, state_type: StateType) -> List[str]:
        """Retrieve relevant patterns based on text and expected state type"""
        if state_type == StateType.ADAPTIVE:
            return [
                pattern
                for pattern in self.adaptive_patterns
                if any(word in text.lower() for word in pattern.split())
            ]
        else:
            return [
                pattern
                for pattern in self.maladaptive_patterns
                if any(word in text.lower() for word in pattern.split())
            ]
