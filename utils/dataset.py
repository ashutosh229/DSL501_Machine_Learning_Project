from typing import List, Dict, Any


def create_sample_dataset() -> List[Dict[str, Any]]:
    """Create sample dataset for testing"""
    sample_posts = [
        {
            "post_id": "post_001",
            "user_id": "user_001",
            "timeline_position": 1,
            "text": "I've been really struggling with depression lately. But I'm trying to get help and talk to a therapist. Some days are really hard and I feel worthless. Other days I feel like I'm making progress and learning to cope better.",
        },
        {
            "post_id": "post_002",
            "user_id": "user_002",
            "timeline_position": 1,
            "text": "I hate myself and everything I do. Nothing ever goes right and I'm just a complete failure. I don't see the point in trying anymore.",
        },
        {
            "post_id": "post_003",
            "user_id": "user_003",
            "timeline_position": 1,
            "text": "Started going to therapy this week. It's scary but I think it's the right step. My therapist seems really understanding and I'm hopeful this will help me work through my issues.",
        },
    ]

    return sample_posts


def create_sample_ground_truth() -> List[Dict[str, Any]]:
    """Create sample ground truth for evaluation"""
    ground_truth = [
        {
            "post_id": "post_001",
            "adaptive_evidence": [
                "I'm trying to get help and talk to a therapist",
                "I feel like I'm making progress and learning to cope better",
            ],
            "maladaptive_evidence": [
                "I've been really struggling with depression",
                "Some days are really hard and I feel worthless",
            ],
        },
        {
            "post_id": "post_002",
            "adaptive_evidence": [],
            "maladaptive_evidence": [
                "I hate myself and everything I do",
                "Nothing ever goes right and I'm just a complete failure",
                "I don't see the point in trying anymore",
            ],
        },
        {
            "post_id": "post_003",
            "adaptive_evidence": [
                "Started going to therapy this week",
                "I think it's the right step",
                "My therapist seems really understanding",
                "I'm hopeful this will help me work through my issues",
            ],
            "maladaptive_evidence": ["It's scary"],
        },
    ]

    return ground_truth
