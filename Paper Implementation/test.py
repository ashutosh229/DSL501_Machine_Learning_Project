import json
from pathlib import Path
import random

# Create data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Helper function to generate a mock post
def generate_post(post_num):
    mood = random.choice(["good", "bad", "anxious", "hopeful", "sad", "motivated"])
    adaptive_phrases = []
    maladaptive_phrases = []
    
    if mood in ["good", "hopeful", "motivated"]:
        adaptive_phrases.append(f"I felt {mood} today and took positive actions.")
        maladaptive_phrases = []
    else:
        maladaptive_phrases.append(f"I felt {mood} and struggled with daily tasks.")
        adaptive_phrases = []
    
    text = f"Post {post_num}: Today I felt {mood}. " + \
           ("I did some productive work and went for a walk." if adaptive_phrases else \
            "I had a hard time getting out of bed and felt overwhelmed.")
    
    post = {
        "post_id": f"post_{post_num}",
        "text": text,
        "adaptive_evidence": adaptive_phrases,
        "maladaptive_evidence": maladaptive_phrases,
        "summary": f"A short summary of post {post_num}",
        "wellbeing_score": 1.0 if adaptive_phrases else 0.0
    }
    return post

# Generate 30 user timelines
for user_num in range(1, 31):
    posts = [generate_post(i) for i in range((user_num-1)*10+1, user_num*10+1)]
    
    timeline = {
        "summary": f"Timeline summary for user_{user_num}",
        "posts": posts
    }
    
    with open(data_folder / f"user_{user_num:03d}.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

print("✅ Generated 30 JSON files with 10 posts each in the 'data/' folder.")
