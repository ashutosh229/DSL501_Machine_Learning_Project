import json
import re
import spacy
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import wandb
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
from models.states import SelfState, ProcessedPost, ClassificationResult
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)







class LLMInterface:
    """Interface for Gemma 2 9B model with 4-bit quantization"""
    
    def __init__(self):
        self.model_name = "google/gemma-2-9b-it"
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load 4-bit quantized Gemma 2 9B model"""
        try:
            logger.info("Loading Gemma 2 9B model with 4-bit quantization...")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to CPU-only mode or mock responses for testing
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from the model"""
        if self.model is None:
            # Mock response for testing when model isn't available
            return "Mock response - model not loaded"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response.replace(prompt, "").strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"

class DataProcessingAgent:
    """Agent 1: Handles data preprocessing and preparation"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        logger.info("Data Processing Agent initialized")
    
    def process_post(self, post_data: Dict[str, Any]) -> ProcessedPost:
        """Process raw Reddit post data"""
        start_time = datetime.now()
        
        # Extract text and metadata
        text = post_data.get('text', '')
        user_id = post_data.get('user_id', 'unknown')
        post_id = post_data.get('post_id', 'unknown')
        timeline_pos = post_data.get('timeline_position', 0)
        
        # Privacy protection - basic anonymization
        text = self._anonymize_text(text)
        
        # Sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        # Create context windows (2-3 sentence chunks)
        context_windows = self._create_context_windows(sentences)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processed post {post_id} in {processing_time:.2f}s")
        
        return ProcessedPost(
            original_text=text,
            sentences=sentences,
            context_windows=context_windows,
            user_id=user_id,
            post_id=post_id,
            timeline_position=timeline_pos
        )
    
    def _anonymize_text(self, text: str) -> str:
        """Basic privacy protection"""
        # Remove usernames, emails, phone numbers
        text = re.sub(r'@\w+', '[USER]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        return text
    
    def _create_context_windows(self, sentences: List[str], window_size: int = 3) -> List[str]:
        """Create overlapping context windows"""
        windows = []
        for i in range(len(sentences)):
            start = max(0, i - window_size // 2)
            end = min(len(sentences), start + window_size)
            window = " ".join(sentences[start:end])
            windows.append(window)
        return windows

class StateFinderAgent:
    """Agent 2: Finds potential self-states using ReAct reasoning"""
    
    def __init__(self, llm: LLMInterface, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb
        logger.info("State Finder Agent initialized")
    
    def find_self_states(self, processed_post: ProcessedPost) -> List[Dict[str, Any]]:
        """Find potential self-states in the processed post"""
        candidates = []
        
        for i, window in enumerate(processed_post.context_windows):
            # ReAct reasoning approach
            thought = self._think_about_window(window)
            action = self._identify_potential_states(window, thought)
            observation = self._observe_results(action, window)
            
            if observation['has_self_state']:
                candidates.append({
                    'text': window,
                    'window_index': i,
                    'potential_spans': observation['spans'],
                    'reasoning': f"Thought: {thought}\nAction: {action}\nObservation: {observation}"
                })
        
        logger.info(f"Found {len(candidates)} potential self-state candidates")
        return candidates
    
    def _think_about_window(self, window: str) -> str:
        """ReAct: Think about what might be in this window"""
        prompt = f"""
        Think about this text and identify potential self-state indicators:
        
        Text: "{window}"
        
        Consider the MIND framework dimensions:
        - Affect: emotional expressions
        - Behavior: actions toward self/others
        - Cognition: thoughts about self/others
        - Desire: expressed needs/fears
        
        What potential self-states might be present?
        """
        
        return self.llm.generate_response(prompt, max_length=256)
    
    def _identify_potential_states(self, window: str, thought: str) -> str:
        """ReAct: Take action to identify states"""
        prompt = f"""
        Based on this analysis: {thought}
        
        Text: "{window}"
        
        Identify specific phrases or sentences that might indicate self-states.
        Look for both obvious and subtle indicators.
        """
        
        return self.llm.generate_response(prompt, max_length=256)
    
    def _observe_results(self, action: str, window: str) -> Dict[str, Any]:
        """ReAct: Observe and evaluate the identification results"""
        # Simple heuristic-based observation for now
        has_self_state = any(keyword in window.lower() for keyword in 
                           ['feel', 'think', 'want', 'need', 'hope', 'fear', 'i am', 'i was'])
        
        spans = []
        if has_self_state:
            # Extract sentences that likely contain self-states
            sentences = window.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in 
                      ['feel', 'think', 'want', 'need', 'hope', 'fear']):
                    spans.append(sentence.strip())
        
        return {
            'has_self_state': has_self_state,
            'spans': spans,
            'confidence': 0.7 if has_self_state else 0.1
        }

class AdaptiveClassifierAgent:
    """Agent 3: Specialized for detecting adaptive self-states"""
    
    def __init__(self, llm: LLMInterface, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb
        self.sensitivity = 0.6  # Higher sensitivity for adaptive states
        logger.info("Adaptive Classifier Agent initialized")
    
    def classify_adaptive(self, candidates: List[Dict[str, Any]]) -> List[SelfState]:
        """Classify candidates as adaptive self-states"""
        adaptive_states = []
        
        for candidate in candidates:
            for span in candidate['potential_spans']:
                if self._is_adaptive_state(span, candidate['text']):
                    confidence = self._calculate_adaptive_confidence(span)
                    reasoning = self._generate_adaptive_reasoning(span)
                    
                    if confidence >= self.sensitivity:
                        adaptive_states.append(SelfState(
                            text=span,
                            state_type=StateType.ADAPTIVE,
                            confidence=confidence,
                            reasoning=reasoning,
                            span_start=candidate['text'].find(span),
                            span_end=candidate['text'].find(span) + len(span)
                        ))
        
        logger.info(f"Classified {len(adaptive_states)} adaptive states")
        return adaptive_states
    
    def _is_adaptive_state(self, span: str, context: str) -> bool:
        """Determine if span represents adaptive state"""
        adaptive_indicators = [
            'help', 'support', 'better', 'improve', 'plan', 'goal',
            'grateful', 'thankful', 'progress', 'learn', 'grow',
            'cope', 'manage', 'handle', 'overcome', 'resilient'
        ]
        
        return any(indicator in span.lower() for indicator in adaptive_indicators)
    
    def _calculate_adaptive_confidence(self, span: str) -> float:
        """Calculate confidence for adaptive classification"""
        # Get relevant patterns from knowledge base
        patterns = self.kb.get_relevant_patterns(span, StateType.ADAPTIVE)
        
        base_confidence = 0.5
        pattern_boost = len(patterns) * 0.1
        
        # Boost confidence for subtle adaptive indicators
        subtle_indicators = ['trying', 'working on', 'hoping', 'planning']
        subtle_boost = sum(0.05 for indicator in subtle_indicators if indicator in span.lower())
        
        return min(0.95, base_confidence + pattern_boost + subtle_boost)
    
    def _generate_adaptive_reasoning(self, span: str) -> str:
        """Generate reasoning for adaptive classification"""
        prompt = f"""
        Explain why this text indicates an adaptive self-state:
        
        Text: "{span}"
        
        An adaptive self-state reflects flexible, constructive processes that promote well-being.
        Consider: problem-solving, help-seeking, emotional regulation, positive coping.
        """
        
        return self.llm.generate_response(prompt, max_length=128)

class MaladaptiveClassifierAgent:
    """Agent 4: Specialized for detecting maladaptive self-states"""
    
    def __init__(self, llm: LLMInterface, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb
        self.sensitivity = 0.7  # Standard sensitivity for maladaptive states
        logger.info("Maladaptive Classifier Agent initialized")
    
    def classify_maladaptive(self, candidates: List[Dict[str, Any]]) -> List[SelfState]:
        """Classify candidates as maladaptive self-states"""
        maladaptive_states = []
        
        for candidate in candidates:
            for span in candidate['potential_spans']:
                if self._is_maladaptive_state(span, candidate['text']):
                    confidence = self._calculate_maladaptive_confidence(span)
                    reasoning = self._generate_maladaptive_reasoning(span)
                    
                    if confidence >= self.sensitivity:
                        maladaptive_states.append(SelfState(
                            text=span,
                            state_type=StateType.MALADAPTIVE,
                            confidence=confidence,
                            reasoning=reasoning,
                            span_start=candidate['text'].find(span),
                            span_end=candidate['text'].find(span) + len(span)
                        ))
        
        logger.info(f"Classified {len(maladaptive_states)} maladaptive states")
        return maladaptive_states
    
    def _is_maladaptive_state(self, span: str, context: str) -> bool:
        """Determine if span represents maladaptive state"""
        maladaptive_indicators = [
            'hate', 'worthless', 'hopeless', 'useless', 'failure',
            'hurt', 'pain', 'suffer', 'destroy', 'ruin',
            'never', 'always', 'terrible', 'awful', 'horrible'
        ]
        
        return any(indicator in span.lower() for indicator in maladaptive_indicators)
    
    def _calculate_maladaptive_confidence(self, span: str) -> float:
        """Calculate confidence for maladaptive classification"""
        patterns = self.kb.get_relevant_patterns(span, StateType.MALADAPTIVE)
        
        base_confidence = 0.6
        pattern_boost = len(patterns) * 0.15
        
        # Boost for explicit negative terms
        explicit_terms = ['hate myself', 'want to die', 'worthless', 'hopeless']
        explicit_boost = sum(0.1 for term in explicit_terms if term in span.lower())
        
        return min(0.95, base_confidence + pattern_boost + explicit_boost)
    
    def _generate_maladaptive_reasoning(self, span: str) -> str:
        """Generate reasoning for maladaptive classification"""
        prompt = f"""
        Explain why this text indicates a maladaptive self-state:
        
        Text: "{span}"
        
        A maladaptive self-state reflects rigid, harmful processes that hinder adaptation.
        Consider: self-criticism, hopelessness, destructive behaviors, emotional dysregulation.
        """
        
        return self.llm.generate_response(prompt, max_length=128)

class ValidationAgent:
    """Agent 5: Validates and cross-checks classification results"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.min_confidence = 0.6
        logger.info("Validation Agent initialized")
    
    def validate_results(self, adaptive_states: List[SelfState], 
                        maladaptive_states: List[SelfState],
                        processed_post: ProcessedPost) -> Dict[str, Any]:
        """Validate classification results"""
        
        # Cross-validation checks
        consistency_score = self._check_consistency(adaptive_states, maladaptive_states)
        confidence_distribution = self._analyze_confidence_distribution(
            adaptive_states + maladaptive_states
        )
        temporal_consistency = self._check_temporal_consistency(
            adaptive_states + maladaptive_states, processed_post
        )
        
        # Overall validation score
        overall_score = (consistency_score + confidence_distribution + temporal_consistency) / 3
        
        validation_result = {
            'is_valid': overall_score >= self.min_confidence,
            'overall_score': overall_score,
            'consistency_score': consistency_score,
            'confidence_score': confidence_distribution,
            'temporal_score': temporal_consistency,
            'flagged_issues': self._identify_issues(adaptive_states, maladaptive_states),
            'recommendation': 'accept' if overall_score >= self.min_confidence else 'refine'
        }
        
        logger.info(f"Validation complete. Score: {overall_score:.2f}, Valid: {validation_result['is_valid']}")
        return validation_result
    
    def _check_consistency(self, adaptive: List[SelfState], maladaptive: List[SelfState]) -> float:
        """Check consistency between adaptive and maladaptive classifications"""
        if not adaptive and not maladaptive:
            return 0.5
        
        total_states = len(adaptive) + len(maladaptive)
        
        # Check for conflicting classifications on overlapping text
        conflicts = 0
        for a_state in adaptive:
            for m_state in maladaptive:
                if self._texts_overlap(a_state.text, m_state.text):
                    conflicts += 1
        
        consistency = max(0.0, 1.0 - (conflicts / total_states))
        return consistency
    
    def _texts_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts have significant overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1.intersection(words2))
        return overlap > max(len(words1), len(words2)) * 0.5
    
    def _analyze_confidence_distribution(self, all_states: List[SelfState]) -> float:
        """Analyze confidence score distribution"""
        if not all_states:
            return 0.5
        
        confidences = [state.confidence for state in all_states]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Penalize if too many low-confidence predictions
        low_confidence_count = sum(1 for c in confidences if c < 0.6)
        penalty = (low_confidence_count / len(confidences)) * 0.3
        
        return max(0.0, avg_confidence - penalty)
    
    def _check_temporal_consistency(self, all_states: List[SelfState], 
                                  processed_post: ProcessedPost) -> float:
        """Check temporal consistency (placeholder for timeline analysis)"""
        # For single post, check if states make sense together
        if len(all_states) <= 1:
            return 0.8
        
        # Simple heuristic: penalize if too many conflicting states
        adaptive_count = sum(1 for s in all_states if s.state_type == StateType.ADAPTIVE)
        maladaptive_count = len(all_states) - adaptive_count
        
        if adaptive_count > 0 and maladaptive_count > 0:
            # Mixed states are common in real data
            return 0.7
        else:
            # All one type might be less realistic
            return 0.6
    
    def _identify_issues(self, adaptive: List[SelfState], maladaptive: List[SelfState]) -> List[str]:
        """Identify specific issues with classifications"""
        issues = []
        
        if not adaptive and not maladaptive:
            issues.append("No self-states detected")
        
        if len(adaptive) > 10:
            issues.append("Too many adaptive states detected")
        
        if len(maladaptive) > 10:
            issues.append("Too many maladaptive states detected")
        
        low_conf_adaptive = [s for s in adaptive if s.confidence < 0.5]
        low_conf_maladaptive = [s for s in maladaptive if s.confidence < 0.5]
        
        if len(low_conf_adaptive) > len(adaptive) * 0.5:
            issues.append("Many low-confidence adaptive classifications")
        
        if len(low_conf_maladaptive) > len(maladaptive) * 0.5:
            issues.append("Many low-confidence maladaptive classifications")
        
        return issues

class RefinementAgent:
    """Agent 6: Improves results when validation fails"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.refinement_strategies = {
            'low_adaptive_recall': self._boost_adaptive_sensitivity,
            'high_false_positives': self._increase_thresholds,
            'inconsistent_results': self._resolve_conflicts,
            'low_confidence': self._improve_reasoning'
        }
        logger.info("Refinement Agent initialized")
    
    def refine_classification(self, validation_result: Dict[str, Any],
                            adaptive_states: List[SelfState],
                            maladaptive_states: List[SelfState],
                            candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refine classification based on validation feedback"""
        
        issues = validation_result.get('flagged_issues', [])
        refinement_actions = []
        
        # Determine refinement strategy based on issues
        if 'Many low-confidence adaptive classifications' in issues:
            refined_adaptive = self._boost_adaptive_sensitivity(adaptive_states, candidates)
            refinement_actions.append('boosted_adaptive_sensitivity')
        else:
            refined_adaptive = adaptive_states
        
        if 'Too many adaptive states detected' in issues or 'Too many maladaptive states detected' in issues:
            refined_adaptive, refined_maladaptive = self._increase_thresholds(
                refined_adaptive, maladaptive_states
            )
            refinement_actions.append('increased_thresholds')
        else:
            refined_maladaptive = maladaptive_states
        
        # Check for conflicts
        if validation_result.get('consistency_score', 1.0) < 0.7:
            refined_adaptive, refined_maladaptive = self._resolve_conflicts(
                refined_adaptive, refined_maladaptive
            )
            refinement_actions.append('resolved_conflicts')
        
        logger.info(f"Applied refinement actions: {refinement_actions}")
        
        return {
            'adaptive_states': refined_adaptive,
            'maladaptive_states': refined_maladaptive,
            'actions_taken': refinement_actions,
            'refinement_confidence': 0.8
        }
    
    def _boost_adaptive_sensitivity(self, adaptive_states: List[SelfState], 
                                  candidates: List[Dict[str, Any]]) -> List[SelfState]:
        """Boost sensitivity for adaptive state detection"""
        boosted_states = list(adaptive_states)  # Keep existing states
        
        # Look for additional adaptive states in candidates with lower threshold
        for candidate in candidates:
            for span in candidate['potential_spans']:
                # Look for subtle adaptive indicators
                subtle_adaptive = [
                    'trying', 'working', 'hoping', 'planning', 'learning',
                    'getting help', 'talking to', 'reaching out'
                ]
                
                if any(indicator in span.lower() for indicator in subtle_adaptive):
                    # Check if not already classified
                    if not any(span in state.text for state in boosted_states):
                        boosted_states.append(SelfState(
                            text=span,
                            state_type=StateType.ADAPTIVE,
                            confidence=0.6,  # Lower threshold
                            reasoning="Detected through sensitivity boost - subtle adaptive indicator",
                            span_start=candidate['text'].find(span),
                            span_end=candidate['text'].find(span) + len(span)
                        ))
        
        return boosted_states
    
    def _increase_thresholds(self, adaptive_states: List[SelfState], 
                           maladaptive_states: List[SelfState]) -> Tuple[List[SelfState], List[SelfState]]:
        """Increase confidence thresholds to reduce false positives"""
        high_threshold = 0.8
        
        refined_adaptive = [s for s in adaptive_states if s.confidence >= high_threshold]
        refined_maladaptive = [s for s in maladaptive_states if s.confidence >= high_threshold]
        
        return refined_adaptive, refined_maladaptive
    
    def _resolve_conflicts(self, adaptive_states: List[SelfState], 
                          maladaptive_states: List[SelfState]) -> Tuple[List[SelfState], List[SelfState]]:
        """Resolve conflicting classifications"""
        resolved_adaptive = []
        resolved_maladaptive = []
        
        for a_state in adaptive_states:
            conflicting = False
            for m_state in maladaptive_states:
                if self._texts_overlap(a_state.text, m_state.text):
                    # Keep the higher confidence classification
                    if a_state.confidence > m_state.confidence:
                        resolved_adaptive.append(a_state)
                    conflicting = True
                    break
            
            if not conflicting:
                resolved_adaptive.append(a_state)
        
        for m_state in maladaptive_states:
            conflicting = False
            for a_state in adaptive_states:
                if self._texts_overlap(a_state.text, m_state.text):
                    if m_state.confidence > a_state.confidence:
                        resolved_maladaptive.append(m_state)
                    conflicting = True
                    break
            
            if not conflicting:
                resolved_maladaptive.append(m_state)
        
        return resolved_adaptive, resolved_maladaptive
    
    def _texts_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts have significant overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1.intersection(words2))
        return overlap > max(len(words1), len(words2)) * 0.5
    
    def _improve_reasoning(self, states: List[SelfState]) -> List[SelfState]:
        """Improve reasoning for low-confidence states"""
        # This would involve re-prompting the LLM for better explanations
        # For now, just boost confidence slightly for states with good patterns
        improved_states = []
        
        for state in states:
            if state.confidence < 0.6:
                # Boost confidence if text contains strong indicators
                strong_indicators = ['definitely', 'really', 'very', 'extremely']
                if any(indicator in state.text.lower() for indicator in strong_indicators):
                    state.confidence = min(0.8, state.confidence + 0.2)
            
            improved_states.append(state)
        
        return improved_states

class AgenticOrchestrator:
    """Main orchestrator that coordinates all agents"""
    
    def __init__(self, use_wandb: bool = False):
        # Initialize components
        self.knowledge_base = KnowledgeBase()
        self.llm = LLMInterface()
        
        # Initialize agents
        self.data_processor = DataProcessingAgent()
        self.state_finder = StateFinderAgent(self.llm, self.knowledge_base