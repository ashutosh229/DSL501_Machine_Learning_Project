from models.timeline import Timeline
from models.redditPost import Post
from typing import List
import json  
from pathlib import Path  

class DataLoader:
    @staticmethod
    def load_timeline(filepath: str) -> Timeline:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timeline = Timeline(
            user_id=Path(filepath).stem,
            summary=data.get('summary', '')
        )
        
        for post_data in data.get('posts', []):
            post = Post(
                post_id=post_data.get('post_id', ''),
                text=post_data.get('text', ''),
                adaptive_evidence=post_data.get('adaptive_evidence', []),
                maladaptive_evidence=post_data.get('maladaptive_evidence', []),
                summary=post_data.get('summary', ''),
                wellbeing_score=post_data.get('wellbeing_score', 0.0)
            )
            timeline.posts.append(post)
        
        return timeline
    
    @staticmethod
    def load_all_timelines(directory: str) -> List[Timeline]:
        timelines = []
        for filepath in Path(directory).glob('*.json'):
            timelines.append(DataLoader.load_timeline(str(filepath)))
        return timelines