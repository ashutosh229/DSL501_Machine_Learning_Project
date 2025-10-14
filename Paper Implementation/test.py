"""
Synthetic CLPsych 2025 Data Generator
Generates 30 realistic timeline JSON files with mental health content
Each timeline contains 10-15 posts with adaptive/maladaptive evidence annotations
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import os

# Seed for reproducibility
random.seed(42)

class SyntheticDataGenerator:
    """Generate realistic mental health timeline data"""
    
    def __init__(self):
        # Template post texts with markers for evidence
        self.post_templates = {
            'depression_seeking_help': [
                "I've been struggling with depression for weeks now. {MALADAPTIVE1} {MALADAPTIVE2} But yesterday I finally {ADAPTIVE1} and scheduled an appointment. {ADAPTIVE2}",
                "Everything feels hopeless lately. {MALADAPTIVE1} I can barely get myself out of bed most days. {MALADAPTIVE2} Today though, I managed to {ADAPTIVE1} which is a small victory.",
                "The depression is getting worse. {MALADAPTIVE1} {MALADAPTIVE2} I know I need help so I {ADAPTIVE1} My therapist suggested {ADAPTIVE2}",
            ],
            'anxiety_coping': [
                "My anxiety has been through the roof. {MALADAPTIVE1} I had a panic attack at work yesterday. {MALADAPTIVE2} But I've been practicing the breathing exercises {ADAPTIVE1} and it's starting to help. {ADAPTIVE2}",
                "I feel anxious all the time. {MALADAPTIVE1} {MALADAPTIVE2} Started using the coping strategies from therapy. {ADAPTIVE1} I'm trying to be patient with myself. {ADAPTIVE2}",
                "The constant worry is exhausting. {MALADAPTIVE1} I can't focus on anything. {MALADAPTIVE2} My therapist recommended meditation so I {ADAPTIVE1} and {ADAPTIVE2}",
            ],
            'social_isolation': [
                "I've been isolating myself from everyone. {MALADAPTIVE1} {MALADAPTIVE2} But today I reached out to an old friend {ADAPTIVE1} and we made plans to meet up. {ADAPTIVE2}",
                "I feel so alone. {MALADAPTIVE1} Nobody understands what I'm going through. {MALADAPTIVE2} Decided to join a support group. {ADAPTIVE1} Met some people who actually get it. {ADAPTIVE2}",
                "Haven't left my apartment in days. {MALADAPTIVE1} {MALADAPTIVE2} Forced myself to go for a walk today {ADAPTIVE1} and called my sister. {ADAPTIVE2}",
            ],
            'self_harm_recovery': [
                "I relapsed again last night. {MALADAPTIVE1} I hate myself for it. {MALADAPTIVE2} Called my therapist this morning {ADAPTIVE1} and we talked through it. {ADAPTIVE2}",
                "The urges to self-harm are strong. {MALADAPTIVE1} {MALADAPTIVE2} Using the distraction techniques I learned. {ADAPTIVE1} Reached out to my crisis support instead. {ADAPTIVE2}",
                "I feel like such a failure. {MALADAPTIVE1} Can't stop thinking about hurting myself. {MALADAPTIVE2} But I remembered my safety plan {ADAPTIVE1} and texted my sponsor. {ADAPTIVE2}",
            ],
            'relationship_struggles': [
                "My partner and I have been fighting constantly. {MALADAPTIVE1} I feel like I'm ruining everything. {MALADAPTIVE2} We decided to try couples therapy. {ADAPTIVE1} Had our first session and it went better than expected. {ADAPTIVE2}",
                "I push everyone away when I'm depressed. {MALADAPTIVE1} {MALADAPTIVE2} Trying to communicate better with my partner. {ADAPTIVE1} I apologized and explained what I'm going through. {ADAPTIVE2}",
                "Feel like I don't deserve to be loved. {MALADAPTIVE1} {MALADAPTIVE2} My partner has been incredibly supportive. {ADAPTIVE1} Learning to accept help and love. {ADAPTIVE2}",
            ],
            'work_stress': [
                "Work is overwhelming me. {MALADAPTIVE1} I'm on the verge of a breakdown. {MALADAPTIVE2} Talked to my manager about reducing my workload. {ADAPTIVE1} Taking a mental health day tomorrow. {ADAPTIVE2}",
                "Can't handle the pressure anymore. {MALADAPTIVE1} {MALADAPTIVE2} Started setting better boundaries at work. {ADAPTIVE1} Learned to say no to extra projects. {ADAPTIVE2}",
                "I'm going to get fired, I know it. {MALADAPTIVE1} My performance has been terrible. {MALADAPTIVE2} Met with HR about accommodations. {ADAPTIVE1} They're being supportive. {ADAPTIVE2}",
            ],
            'medication_journey': [
                "The new medication makes me feel numb. {MALADAPTIVE1} I don't feel like myself anymore. {MALADAPTIVE2} Scheduled an appointment to discuss adjusting the dosage. {ADAPTIVE1} Keeping a mood journal to track changes. {ADAPTIVE2}",
                "Side effects are awful. {MALADAPTIVE1} {MALADAPTIVE2} Sticking with it because my doctor said it takes time. {ADAPTIVE1} Noticing small improvements in my mood. {ADAPTIVE2}",
                "Thinking about stopping my meds. {MALADAPTIVE1} {MALADAPTIVE2} Called my psychiatrist first. {ADAPTIVE1} We're trying a different medication instead. {ADAPTIVE2}",
            ],
            'progress_setback': [
                "Thought I was getting better but I'm not. {MALADAPTIVE1} Back to square one. {MALADAPTIVE2} Reminded myself that recovery isn't linear. {ADAPTIVE1} Reaching out for support instead of giving up. {ADAPTIVE2}",
                "Had a really bad day after weeks of progress. {MALADAPTIVE1} {MALADAPTIVE2} Using my coping skills instead of old habits. {ADAPTIVE1} This setback doesn't erase my progress. {ADAPTIVE2}",
                "Feel like I'll never get better. {MALADAPTIVE1} All this therapy for nothing. {MALADAPTIVE2} My therapist reminded me how far I've come. {ADAPTIVE1} Looking at my progress journal from six months ago. {ADAPTIVE2}",
            ],
            'daily_struggles': [
                "Can't even do basic tasks today. {MALADAPTIVE1} {MALADAPTIVE2} Managed to shower and eat breakfast. {ADAPTIVE1} Small steps are still steps forward. {ADAPTIVE2}",
                "Everything feels impossible. {MALADAPTIVE1} I'm worthless. {MALADAPTIVE2} Did one small thing today - made my bed. {ADAPTIVE1} Trying to celebrate tiny victories. {ADAPTIVE2}",
                "Another day of feeling terrible. {MALADAPTIVE1} {MALADAPTIVE2} Went outside for 10 minutes. {ADAPTIVE1} Called a friend. {ADAPTIVE2}",
            ],
            'positive_progress': [
                "Starting to see the light at the end of the tunnel. {ADAPTIVE1} Therapy is really helping. {ADAPTIVE2} Still have rough days. {MALADAPTIVE1} But they're getting less frequent. {MALADAPTIVE2}",
                "Feeling hopeful for the first time in months. {ADAPTIVE1} {ADAPTIVE2} Had a bad moment yesterday. {MALADAPTIVE1} But I worked through it using my tools. {ADAPTIVE2}",
                "Three weeks without self-harming. {ADAPTIVE1} So proud of myself. {ADAPTIVE2} Almost slipped up. {MALADAPTIVE1} But I reached out for help instead. {ADAPTIVE2}",
            ],
        }
        
        # Evidence phrases
        self.maladaptive_phrases = {
            'MALADAPTIVE1': [
                "I feel completely hopeless",
                "I want to give up on everything",
                "I hate myself so much",
                "I'm worthless and pathetic",
                "Nothing will ever get better",
                "I can't do anything right",
                "Everyone would be better off without me",
                "I'm a burden to everyone",
                "I feel empty inside",
                "I'm tired of existing",
                "I have no reason to keep going",
                "I'm broken beyond repair",
                "I deserve to suffer",
                "I'm disgusting",
                "Life is meaningless",
            ],
            'MALADAPTIVE2': [
                "I've been having suicidal thoughts",
                "I isolated myself from all my friends",
                "I haven't showered in a week",
                "I stopped taking my medication",
                "I've been drinking to cope",
                "I can't get out of bed anymore",
                "I'm failing at everything",
                "I hurt myself again",
                "I pushed everyone away",
                "I'm completely numb",
                "I gave up trying",
                "I don't care about anything anymore",
                "I'm spiraling out of control",
                "I can't see a future for myself",
                "I'm losing my mind",
            ]
        }
        
        self.adaptive_phrases = {
            'ADAPTIVE1': [
                "reached out to my therapist",
                "called the crisis hotline",
                "talked to a friend about what I'm going through",
                "started going to therapy",
                "joined a support group",
                "practiced my coping strategies",
                "used my DBT skills",
                "went for a walk to clear my head",
                "took my medication as prescribed",
                "reached out for help",
                "wrote in my journal",
                "did some self-care activities",
                "set healthy boundaries",
                "asked for support",
                "tried a new coping mechanism",
            ],
            'ADAPTIVE2': [
                "I'm learning to be kinder to myself",
                "Recovery is a journey, not a destination",
                "I'm proud of myself for trying",
                "Taking it one day at a time",
                "I deserve to get better",
                "I'm not giving up",
                "Small steps are still progress",
                "I'm worth the effort",
                "Things can get better",
                "I'm building a support system",
                "I'm learning healthier coping skills",
                "I'm working on self-compassion",
                "I'm committed to my recovery",
                "I'm being patient with myself",
                "I acknowledge my progress",
            ]
        }
    
    def generate_post_content(self, template_category):
        """Generate a post with realistic evidence annotations"""
        template = random.choice(self.post_templates[template_category])
        
        # Replace markers with actual phrases
        text = template
        evidence_map = {}
        
        for marker in ['MALADAPTIVE1', 'MALADAPTIVE2', 'ADAPTIVE1', 'ADAPTIVE2']:
            if marker in text:
                phrase_type = 'maladaptive_phrases' if 'MALADAPTIVE' in marker else 'adaptive_phrases'
                phrase = random.choice(getattr(self, phrase_type)[marker])
                evidence_map[marker] = phrase
                text = text.replace(f'{{{marker}}}', phrase)
        
        # Extract evidence
        adaptive_evidence = [evidence_map.get('ADAPTIVE1', ''), evidence_map.get('ADAPTIVE2', '')]
        adaptive_evidence = [e for e in adaptive_evidence if e]
        
        maladaptive_evidence = [evidence_map.get('MALADAPTIVE1', ''), evidence_map.get('MALADAPTIVE2', '')]
        maladaptive_evidence = [e for e in maladaptive_evidence if e]
        
        return text, adaptive_evidence, maladaptive_evidence
    
    def generate_timeline_summary(self, timeline_id):
        """Generate a summary for the timeline"""
        summaries = [
            f"User timeline_{timeline_id} documents a journey through depression and anxiety, showing gradual progress with therapy and medication.",
            f"Timeline_{timeline_id} chronicles struggles with self-harm and the path to recovery through support groups and therapy.",
            f"User timeline_{timeline_id} describes dealing with social anxiety and isolation, with increasing engagement in treatment.",
            f"Timeline_{timeline_id} shows someone managing bipolar disorder with medication adjustments and therapeutic support.",
            f"User timeline_{timeline_id} documents recovery from severe depression, including setbacks and ultimate progress.",
            f"Timeline_{timeline_id} follows a journey through PTSD treatment with increasing use of coping strategies.",
            f"User timeline_{timeline_id} chronicles struggles with OCD and anxiety, showing gradual improvement with therapy.",
            f"Timeline_{timeline_id} documents managing panic disorder and agoraphobia with professional help.",
        ]
        return random.choice(summaries)
    
    def generate_post_summary(self, adaptive_count, maladaptive_count):
        """Generate a summary for the post"""
        if maladaptive_count > adaptive_count:
            return "Post primarily shows distress with some coping attempts"
        elif adaptive_count > maladaptive_count:
            return "Post shows active coping and progress despite challenges"
        else:
            return "Post shows mixed emotional state with both struggles and coping"
    
    def calculate_wellbeing_score(self, adaptive_count, maladaptive_count):
        """Calculate a wellbeing score based on evidence"""
        if adaptive_count + maladaptive_count == 0:
            return 0.5
        
        score = adaptive_count / (adaptive_count + maladaptive_count)
        # Add some noise
        score += random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, score))
    
    def generate_timeline(self, timeline_id, num_posts=None):
        """Generate a complete timeline with posts"""
        if num_posts is None:
            num_posts = random.randint(10, 15)
        
        timeline = {
            "summary": self.generate_timeline_summary(timeline_id),
            "posts": []
        }
        
        # Generate posts
        categories = list(self.post_templates.keys())
        for post_idx in range(num_posts):
            # Select category with some variety
            category = random.choice(categories)
            
            text, adaptive_evidence, maladaptive_evidence = self.generate_post_content(category)
            
            post = {
                "post_id": f"timeline_{timeline_id}_post_{post_idx+1}",
                "text": text,
                "adaptive_evidence": adaptive_evidence,
                "maladaptive_evidence": maladaptive_evidence,
                "summary": self.generate_post_summary(len(adaptive_evidence), len(maladaptive_evidence)),
                "wellbeing_score": round(self.calculate_wellbeing_score(len(adaptive_evidence), len(maladaptive_evidence)), 2)
            }
            
            timeline["posts"].append(post)
        
        return timeline
    
    def generate_all_timelines(self, output_dir="clpsych_data", num_timelines=30):
        """Generate all 30 timeline JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating {num_timelines} timeline files...")
        print(f"Output directory: {output_path.absolute()}")
        
        total_posts = 0
        
        for timeline_id in range(1, num_timelines + 1):
            timeline = self.generate_timeline(timeline_id)
            total_posts += len(timeline["posts"])
            
            filename = output_path / f"timeline_{timeline_id:03d}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(timeline, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Created {filename.name} with {len(timeline['posts'])} posts")
        
        print(f"\n{'='*60}")
        print(f"Generation Complete!")
        print(f"{'='*60}")
        print(f"Total timelines: {num_timelines}")
        print(f"Total posts: {total_posts}")
        print(f"Average posts per timeline: {total_posts/num_timelines:.1f}")
        print(f"Output directory: {output_path.absolute()}")
        
        # Generate statistics
        self.generate_statistics(output_path)
    
    def generate_statistics(self, output_dir):
        """Generate statistics about the generated data"""
        stats = {
            "total_timelines": 0,
            "total_posts": 0,
            "total_adaptive_spans": 0,
            "total_maladaptive_spans": 0,
            "avg_adaptive_per_post": 0,
            "avg_maladaptive_per_post": 0,
            "avg_wellbeing_score": 0,
        }
        
        wellbeing_scores = []
        adaptive_counts = []
        maladaptive_counts = []
        
        for filepath in Path(output_dir).glob("timeline_*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                timeline = json.load(f)
                stats["total_timelines"] += 1
                
                for post in timeline["posts"]:
                    stats["total_posts"] += 1
                    adaptive_count = len(post["adaptive_evidence"])
                    maladaptive_count = len(post["maladaptive_evidence"])
                    
                    stats["total_adaptive_spans"] += adaptive_count
                    stats["total_maladaptive_spans"] += maladaptive_count
                    
                    adaptive_counts.append(adaptive_count)
                    maladaptive_counts.append(maladaptive_count)
                    wellbeing_scores.append(post["wellbeing_score"])
        
        if stats["total_posts"] > 0:
            stats["avg_adaptive_per_post"] = stats["total_adaptive_spans"] / stats["total_posts"]
            stats["avg_maladaptive_per_post"] = stats["total_maladaptive_spans"] / stats["total_posts"]
            stats["avg_wellbeing_score"] = sum(wellbeing_scores) / len(wellbeing_scores)
        
        # Save statistics
        stats_file = Path(output_dir) / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total timelines: {stats['total_timelines']}")
        print(f"Total posts: {stats['total_posts']}")
        print(f"Total adaptive evidence spans: {stats['total_adaptive_spans']}")
        print(f"Total maladaptive evidence spans: {stats['total_maladaptive_spans']}")
        print(f"Average adaptive spans per post: {stats['avg_adaptive_per_post']:.2f}")
        print(f"Average maladaptive spans per post: {stats['avg_maladaptive_per_post']:.2f}")
        print(f"Average wellbeing score: {stats['avg_wellbeing_score']:.2f}")
        print(f"\nStatistics saved to: {stats_file}")


def main():
    """Main execution"""
    print("="*60)
    print("CLPsych 2025 Synthetic Data Generator")
    print("="*60)
    print()
    
    generator = SyntheticDataGenerator()
    generator.generate_all_timelines(
        output_dir="data",
        num_timelines=30
    )
    
    print("\n" + "="*60)
    print("Sample Timeline Preview")
    print("="*60)
    
    # Show a sample
    with open("data/timeline_001.json", 'r', encoding='utf-8') as f:
        sample = json.load(f)
    
    print(f"\nTimeline Summary: {sample['summary']}")
    print(f"\nFirst Post:")
    print(f"  ID: {sample['posts'][0]['post_id']}")
    print(f"  Text: {sample['posts'][0]['text']}")
    print(f"  Adaptive Evidence: {sample['posts'][0]['adaptive_evidence']}")
    print(f"  Maladaptive Evidence: {sample['posts'][0]['maladaptive_evidence']}")
    print(f"  Wellbeing Score: {sample['posts'][0]['wellbeing_score']}")
    
    print("\n" + "="*60)
    print("Data Generation Complete!")
    print("="*60)
    print("\nYou can now use this data with:")
    print("  from self_state_classifier import DataLoader")
    print("  timelines = DataLoader.load_all_timelines('data/')")


if __name__ == "__main__":
    main()