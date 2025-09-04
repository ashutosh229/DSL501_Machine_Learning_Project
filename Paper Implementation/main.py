import json
import re
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from bert_score import score as bert_score

# Load spaCy model for sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class SelfStateEvidence:
    """Represents a self-state evidence span"""
    text: str
    label: str  # 'adaptive' or 'maladaptive'
    start_pos: int = -1
    end_pos: int = -1
    confidence: float = 0.0

@dataclass
class RedditPost:
    """Represents a Reddit post with annotations"""
    post_id: str
    text: str
    adaptive_evidence: List[str] = None
    maladaptive_evidence: List[str] = None
    summary: str = ""
    well_being_score: float = 0.0

class SelfStateClassifier(ABC):
    """Abstract base class for self-state classifiers"""
    
    @abstractmethod
    def predict(self, post: RedditPost) -> List[SelfStateEvidence]:
        """Predict self-state evidence in a post"""
        pass

class BaselineSentenceClassifier(SelfStateClassifier):
    """Baseline method: classify each sentence as adaptive/maladaptive"""

    def __init__(self, model_name: str = "google/gemma-2-9b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the quantized Gemma model"""
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using mock responses for demonstration")
            self.model = None
            self.tokenizer = None

    def _get_baseline_prompt(self, post: RedditPost) -> str:
        """Generate baseline prompt for sentence classification"""
        return f""" You are a professional psychologist .
,→ Given a social media post ,
,→ classify whether or not a
,→ sentence demonstrates an adaptive
,→ or maladaptive self - state .
An adaptive self - state reflects aspects
,→ of the self that are flexible ,
,→ non - ruminative , and promote well -
,→ being and optimal functioning .
A maladaptive self - state reflects
,→ internal states or perspectives
,→ that hinder an individual 's
,→ ability to adapt to situations or
,→ cope with challenges effectively
,→ , potentially leading to
,→ emotional distress or behavioral
,→ problems .
Here is the sentence :
{ post.text }"""

    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using spaCy"""
        if nlp is None:
            # Fallback sentence segmentation
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _classify_sentence(self, sentence: str, context: str = "") -> str:
        """Classify a single sentence"""
        if self.model is None:
            # Mock classification for demonstration
            negative_indicators = ['sad', 'depressed', 'hopeless', 'worthless', 'hate', 'kill', 'die', 'hurt']
            positive_indicators = ['help', 'better', 'hope', 'support', 'care', 'love', 'plan', 'future']

            sentence_lower = sentence.lower()
            neg_score = sum(1 for word in negative_indicators if word in sentence_lower)
            pos_score = sum(1 for word in positive_indicators if word in sentence_lower)

            return 'maladaptive' if neg_score > pos_score else 'adaptive'

        prompt = self._get_baseline_prompt(sentence)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract classification from response
        response_lower = response.lower().strip()
        if 'maladaptive' in response_lower:
            return 'maladaptive'
        elif 'adaptive' in response_lower:
            return 'adaptive'
        else:
            return 'adaptive'  # Default to adaptive

    def predict(self, post: RedditPost) -> List[SelfStateEvidence]:
        """Predict self-state evidence using baseline sentence classification"""
        sentences = self._segment_sentences(post.text)
        evidence = []

        for sentence in sentences:
            if len(sentence.strip()) < 5:  # Skip very short sentences
                continue

            classification = self._classify_sentence(sentence)
            evidence.append(SelfStateEvidence(
                text=sentence,
                label=classification
            ))

        return evidence

class ContextAwareSentenceClassifier(BaselineSentenceClassifier):
    """Context-aware version of baseline classifier"""
    
    def _get_context_prompt(self, sentence: str, context: str) -> str:
        """Generate context-aware prompt"""
        return f"""You are a professional psychologist. Given a social media post, classify whether or not a sentence demonstrates an adaptive or maladaptive self-state.

An adaptive self-state reflects internal processes that are flexible, constructive, and promote emotional well-being, effective functioning, and psychological health.

A maladaptive self-state reflects internal processes that are rigid, ruminative, self-defeating, or harmful, and are often associated with emotional distress or impaired functioning.

To make your classification, use the ABCD framework for psychological self-states:

A. **Affect** Type of emotional expression
- Adaptive: calm, content, assertive, proud, justifiable pain/grief
- Maladaptive: anxious, hopeless, apathetic, aggressive, ashamed, depressed

B. **Behavior** Main behavioral tendencies
- Toward Others (BO):
  - Adaptive: relational, autonomous behavior
  - Maladaptive: fight/flight response, controlling or overcontrolled behavior
- Toward Self (BS):
  - Adaptive: self-care
  - Maladaptive: self-neglect, avoidance, self-harm

C. **Cognition** Main thought patterns
- Toward Others (CO):
  - Adaptive: perceiving others as supportive or related
  - Maladaptive: perceiving others as detached, overattached, or autonomy-blocking
- Toward Self (CS):
  - Adaptive: self-compassion and acceptance
  - Maladaptive: self-criticism

D. **Desire** Expressed needs, goals, intentions, or fears
- Adaptive: desire for autonomy, relatedness, self-esteem, care
- Maladaptive: fear that these needs won't be met

Here is the post so far:
{context}

Here is the current sentence:
{sentence}

Classification (adaptive/maladaptive):"""
    
    def predict(self, post: RedditPost) -> List[SelfStateEvidence]:
        """Predict with context awareness"""
        sentences = self._segment_sentences(post.text)
        evidence = []
        context = ""
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 5:
                context += sentence + " "
                continue
            
            classification = self._classify_sentence_with_context(sentence, context)
            evidence.append(SelfStateEvidence(
                text=sentence,
                label=classification
            ))
            
            context += sentence + " "
        
        return evidence
    
    def _classify_sentence_with_context(self, sentence: str, context: str) -> str:
        """Classify sentence with context"""
        if self.model is None:
            # Enhanced mock classification considering context
            context_lower = context.lower()
            sentence_lower = sentence.lower()
            
            # Context-aware indicators
            if 'but' in sentence_lower or 'however' in sentence_lower:
                # Potential shift in sentiment
                positive_shift = any(word in sentence_lower for word in ['better', 'help', 'hope', 'support'])
                if positive_shift:
                    return 'adaptive'
            
            return self._classify_sentence(sentence, context)
        
        prompt = self._get_context_prompt(sentence, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        response_lower = response.lower().strip()
        if 'maladaptive' in response_lower:
            return 'maladaptive'
        elif 'adaptive' in response_lower:
            return 'adaptive'
        else:
            return 'adaptive'

class ImportanceFilteringClassifier(ContextAwareSentenceClassifier):
    """Classifier with importance filtering preprocessing step"""
    
    def _get_importance_prompt(self, sentence: str) -> str:
        """Generate importance filtering prompt"""
        return f"""You are a professional psychologist. Given a social media post, decide whether or not the sentence is critically important. A sentence is critical if it evidences one of six things: it 1) expresses a distinct emotion (A), 2) expresses a person's interactions with another (B-O), 3) expresses a person's interactions with themselves (B-S), 4) expresses a person's perceptions of another (B-O) 5) expresses a person's perceptions of themselves, (C-O) or 6) expresses an explicit desire, need, intention, fear or expectation. (D) Not every sentence is important. If the sentence is critical, return True. If not, return False.

Now, it's your turn.
Here is how the post starts:
{sentence}

Important (True/False):"""
    
    def _is_sentence_important(self, sentence: str) -> bool:
        """Determine if sentence is important using LLM"""
        if self.model is None:
            # Mock importance filtering
            important_indicators = [
                'feel', 'think', 'want', 'need', 'hope', 'fear', 'angry', 'sad',
                'happy', 'depressed', 'anxious', 'help', 'support', 'alone',
                'together', 'relationship', 'friend', 'family', 'myself', 'i am'
            ]
            
            sentence_lower = sentence.lower()
            return any(indicator in sentence_lower for indicator in important_indicators)
        
        prompt = self._get_importance_prompt(sentence)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return 'true' in response.lower().strip()
    
    def predict(self, post: RedditPost) -> List[SelfStateEvidence]:
        """Predict with importance filtering"""
        sentences = self._segment_sentences(post.text)
        evidence = []
        context = ""
        
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                context += sentence + " "
                continue
            
            # Filter by importance first
            if not self._is_sentence_important(sentence):
                context += sentence + " "
                continue
            
            classification = self._classify_sentence_with_context(sentence, context)
            evidence.append(SelfStateEvidence(
                text=sentence,
                label=classification
            ))
            
            context += sentence + " "
        
        return evidence

class SpanIdentificationClassifier(SelfStateClassifier):
    """LLM span identification method"""
    
    def __init__(self, model_name: str = "google/gemma-2-9b", adaptive_boost: bool = False):
        self.model_name = model_name
        self.adaptive_boost = adaptive_boost
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the quantized Gemma model"""
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using mock responses for demonstration")
            self.model = None
            self.tokenizer = None
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using spaCy"""
        if nlp is None:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _create_chunks(self, sentences: List[str]) -> List[str]:
        """Create 2-sentence chunks"""
        chunks = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                chunks.append(sentences[i] + " " + sentences[i + 1])
            else:
                chunks.append(sentences[i])
        return chunks
    
    def _get_span_identification_prompt(self, chunk: str, context: str) -> str:
        """Generate span identification prompt"""
        base_prompt = f"""You are a professional psychologist. Your task is to analyze the following social media post and identify any phrases or subspans that reflect an **adaptive** or **maladaptive** self-state, even if they are mixed within the same sentence or paragraph.

An adaptive self-state reflects internal processes that are flexible, constructive, and promote emotional well-being, effective functioning, and psychological health. Pay close attention to subtle adaptive self-states within sentences.

A maladaptive self-state reflects internal processes that are rigid, ruminative, self-defeating, or harmful, and are often associated with emotional distress or impaired functioning.

To make your classification, use the ABCD framework for psychological self-states:

A. **Affect** Type of emotional expression
- Adaptive: calm, content, assertive, proud, justifiable pain/grief
- Maladaptive: anxious, hopeless, apathetic, aggressive, ashamed, depressed

B. **Behavior** Main behavioral tendencies
- Toward Others (BO):
  - Adaptive: relational, autonomous behavior
  - Maladaptive: fight/flight response, controlling or overcontrolled behavior
- Toward Self (BS):
  - Adaptive: self-care
  - Maladaptive: self-neglect, avoidance, self-harm

C. **Cognition** Main thought patterns
- Toward Others (CO):
  - Adaptive: perceiving others as supportive or related
  - Maladaptive: perceiving others as detached, overattached, or autonomy-blocking
- Toward Self (CS):
  - Adaptive: self-compassion and acceptance
  - Maladaptive: self-criticism

D. **Desire** Expressed needs, goals, intentions, or fears
- Adaptive: desire for autonomy, relatedness, self-esteem, care
- Maladaptive: fear that these needs won't be met

Here is the context of the post so far:
{context}

Here is the current chunk of the post:
{chunk}

Return your predictions as a list of tuples: [("label", "span_text"), ...]"""
        
        if self.adaptive_boost:
            base_prompt += """

Your output should list any sentences that reflect either state. Sometimes, you will need to highlight a phrase inside a sentence - self-states can be subtle. You may return **multiple** adaptive or maladaptive spans per chunk.

If a sentence seems neutral, mark it as adaptive. Try to annotate as much as possible - you should shoot for the highest recall possible."""
        
        return base_prompt
    
    def _extract_spans_from_chunk(self, chunk: str, context: str) -> List[SelfStateEvidence]:
        """Extract self-state spans from a chunk"""
        if self.model is None:
            # Mock span extraction
            spans = []
            
            # Simple pattern-based extraction for demonstration
            negative_patterns = [
                r'\b(hate|kill|die|hurt|hopeless|worthless|depressed|sad|anxious)\b',
                r'\b(can\'t|cannot|never|nothing|no one|nobody)\b.*\b(help|care|understand)\b'
            ]
            
            positive_patterns = [
                r'\b(help|support|care|love|hope|better|good|plan|future)\b',
                r'\b(thank|grateful|appreciate|happy|proud)\b'
            ]
            
            for pattern in negative_patterns:
                matches = re.finditer(pattern, chunk, re.IGNORECASE)
                for match in matches:
                    spans.append(SelfStateEvidence(
                        text=match.group(),
                        label='maladaptive',
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
            
            for pattern in positive_patterns:
                matches = re.finditer(pattern, chunk, re.IGNORECASE)
                for match in matches:
                    spans.append(SelfStateEvidence(
                        text=match.group(),
                        label='adaptive',
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
            
            return spans
        
        prompt = self._get_span_identification_prompt(chunk, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return self._parse_span_response(response, chunk)
    
    def _parse_span_response(self, response: str, chunk: str) -> List[SelfStateEvidence]:
        """Parse LLM response to extract spans"""
        spans = []
        
        # Try to extract tuples from response
        pattern = r'\("?(adaptive|maladaptive)"?,\s*"([^"]+)"\)'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for label, text in matches:
            if text.strip() in chunk:
                start_pos = chunk.find(text.strip())
                end_pos = start_pos + len(text.strip())
                spans.append(SelfStateEvidence(
                    text=text.strip(),
                    label=label.lower(),
                    start_pos=start_pos,
                    end_pos=end_pos
                ))
        
        return spans
    
    def predict(self, post: RedditPost) -> List[SelfStateEvidence]:
        """Predict using span identification"""
        sentences = self._segment_sentences(post.text)
        chunks = self._create_chunks(sentences)
        all_evidence = []
        context = ""
        
        for chunk in chunks:
            spans = self._extract_spans_from_chunk(chunk, context)
            all_evidence.extend(spans)
            context += chunk + " "
        
        return all_evidence

class EvaluationMetrics:
    """Evaluation metrics for self-state classification"""
    
    @staticmethod
    def calculate_recall(predictions: List[SelfStateEvidence], 
                        gold_evidence: List[str],
                        label_filter: str = None) -> float:
        """Calculate recall using BERTScore"""
        if not gold_evidence or not predictions:
            return 0.0
        
        filtered_preds = predictions
        if label_filter:
            filtered_preds = [p for p in predictions if p.label == label_filter]
        
        if not filtered_preds:
            return 0.0
        
        pred_texts = [p.text for p in filtered_preds]
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score(pred_texts, gold_evidence, lang='en', rescale_with_baseline=True)
            return R.mean().item()
        except:
            # Fallback to simple string matching
            matches = 0
            for pred_text in pred_texts:
                for gold_text in gold_evidence:
                    if pred_text.lower().strip() in gold_text.lower().strip() or \
                       gold_text.lower().strip() in pred_text.lower().strip():
                        matches += 1
                        break
            return matches / len(gold_evidence)
    
    @staticmethod
    def calculate_weighted_recall(predictions: List[SelfStateEvidence], 
                                 gold_evidence: List[str],
                                 label_filter: str = None) -> float:
        """Calculate weighted recall based on token count similarity"""
        if not gold_evidence or not predictions:
            return 0.0
        
        filtered_preds = predictions
        if label_filter:
            filtered_preds = [p for p in predictions if p.label == label_filter]
        
        if not filtered_preds:
            return 0.0
        
        pred_token_count = sum(len(p.text.split()) for p in filtered_preds)
        gold_token_count = sum(len(text.split()) for text in gold_evidence)
        
        if gold_token_count == 0:
            return 0.0
        
        # Weight recall by token count similarity
        token_ratio = min(pred_token_count / gold_token_count, 1.0)
        base_recall = EvaluationMetrics.calculate_recall(predictions, gold_evidence, label_filter)
        
        return base_recall * token_ratio

def load_data(json_files: List[str]) -> List[RedditPost]:
    """Load Reddit timeline data from JSON files"""
    posts = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                timeline_data = json.load(f)
                
            # Extract posts from timeline structure
            if 'timeline' in timeline_data:
                for entry in timeline_data['timeline']:
                    if 'posts' in entry:
                        for post_data in entry['posts']:
                            post = RedditPost(
                                post_id=post_data.get('post_id', ''),
                                text=post_data.get('text', ''),
                                adaptive_evidence=post_data.get('adaptive_evidence', []),
                                maladaptive_evidence=post_data.get('maladaptive_evidence', []),
                                summary=post_data.get('summary', ''),
                                well_being_score=post_data.get('well_being_score', 0.0)
                            )
                            posts.append(post)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return posts

def run_evaluation(classifier: SelfStateClassifier, test_posts: List[RedditPost]) -> Dict[str, float]:
    """Run evaluation on test posts"""
    results = {
        'overall_recall': 0.0,
        'adaptive_recall': 0.0,
        'maladaptive_recall': 0.0,
        'weighted_overall_recall': 0.0,
        'weighted_adaptive_recall': 0.0,
        'weighted_maladaptive_recall': 0.0
    }
    
    all_predictions = []
    all_adaptive_gold = []
    all_maladaptive_gold = []
    
    for post in test_posts:
        predictions = classifier.predict(post)
        all_predictions.extend(predictions)
        
        if post.adaptive_evidence:
            all_adaptive_gold.extend(post.adaptive_evidence)
        if post.maladaptive_evidence:
            all_maladaptive_gold.extend(post.maladaptive_evidence)
    
    # Calculate recalls
    all_gold = all_adaptive_gold + all_maladaptive_gold
    
    if all_gold:
        results['overall_recall'] = EvaluationMetrics.calculate_recall(all_predictions, all_gold)
        results['weighted_overall_recall'] = EvaluationMetrics.calculate_weighted_recall(all_predictions, all_gold)
    
    if all_adaptive_gold:
        results['adaptive_recall'] = EvaluationMetrics.calculate_recall(all_predictions, all_adaptive_gold, 'adaptive')
        results['weighted_adaptive_recall'] = EvaluationMetrics.calculate_weighted_recall(all_predictions, all_adaptive_gold, 'adaptive')
    
    if all_maladaptive_gold:
        results['maladaptive_recall'] = EvaluationMetrics.calculate_recall(all_predictions, all_maladaptive_gold, 'maladaptive')
        results['weighted_maladaptive_recall'] = EvaluationMetrics.calculate_weighted_recall(all_predictions, all_maladaptive_gold, 'maladaptive')
    
    return results

def main():
    """Main function to demonstrate the system"""
    print("Self-State Classification System - CLPsych 2025")
    print("=" * 50)
    
    # Example post for demonstration
    example_post = RedditPost(
        post_id="example_1",
        text="I feel so hopeless lately. Nothing seems to matter anymore. But I'm trying to reach out for help because I know that's what I should do. Maybe talking to someone will make a difference.",
        adaptive_evidence=["I'm trying to reach out for help", "Maybe talking to someone will make a difference"],
        maladaptive_evidence=["I feel so hopeless lately", "Nothing seems to matter anymore"]
    )
    
    # Initialize classifiers
    classifiers = {
        'Baseline': BaselineSentenceClassifier(),
        'Context-Aware': ContextAwareSentenceClassifier(),
        'Importance Filtering': ImportanceFilteringClassifier(),
        'Span Identification': SpanIdentificationClassifier(),
        'Span ID + Adaptive Boost': SpanIdentificationClassifier(adaptive_boost=True)
    }
    
    # Test each classifier
    for name, classifier in classifiers.items():
        print(f"\n{name} Results:")
        print("-" * 30)
        
        predictions = classifier.predict(example_post)
        
        print(f"Number of predictions: {len(predictions)}")
        
        adaptive_preds = [p for p in predictions if p.label == 'adaptive']
        maladaptive_preds = [p for p in predictions if p.label == 'maladaptive']
        
        print(f"Adaptive predictions ({len(adaptive_preds)}):")
        for pred in adaptive_preds:
            print(f"  - '{pred.text[:50]}...' " if len(pred.text) > 50 else f"  - '{pred.text}'")
        
        print(f"Maladaptive predictions ({len(maladaptive_preds)}):")
        for pred in maladaptive_preds:
            print(f"  - '{pred.text[:50]}...' " if len(pred.text) > 50 else f"  - '{pred.text}'")
        
        # Calculate simple evaluation metrics
        results = run_evaluation(classifier, [example_post])
        print(f"Overall Recall: {results['overall_recall']:.3f}")
        print(f"Adaptive Recall: {results['adaptive_recall']:.3f}")
        print(f"Maladaptive Recall: {results['maladaptive_recall']:.3f}")

if __name__ == "__main__":
    main()
