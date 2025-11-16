import spacy
import re 
import json
from scripts.llm.mock import MockLLM
from typing import List, Tuple
from models.selfSateSpan import SelfStateSpan   
from prompts.baselinePrompt import BaselinePrompt  
from prompts.contextPrompt import ContextPrompt  
from prompts.importancePrompt import ImportancePrompt
from prompts.spanIdentificationPrompt import SpanIdentificationPrompt  
from prompts.spanIdentificationAdaptiveBoost import SpanIdentificationAdaptiveBoostPrompt


class SelfStateClassifier:
    
    def __init__(self, llm_interface=None, use_mock: bool = False):
        self.nlp = spacy.load("en_core_web_sm")
        if use_mock or llm_interface is None:
            self.llm = MockLLM()
        else:
            self.llm = llm_interface
        self.llm.load_model()
        
    def split_into_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def baseline_classify(self, post_text: str) -> Tuple[List[SelfStateSpan], List[SelfStateSpan]]:
        sentences = self.split_into_sentences(post_text)
        adaptive_spans = []
        maladaptive_spans = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            prompt = BaselinePrompt.BASELINE_PROMPT.format(sentence=sentence)
            response = self.llm.generate(prompt, max_new_tokens=10)
            
            label = self._parse_classification_response(response)
            
            if label == "adaptive":
                adaptive_spans.append(SelfStateSpan(text=sentence, label="adaptive"))
            elif label == "maladaptive":
                maladaptive_spans.append(SelfStateSpan(text=sentence, label="maladaptive"))
        
        return adaptive_spans, maladaptive_spans
    
    def baseline_with_context_classify(self, post_text: str) -> Tuple[List[SelfStateSpan], List[SelfStateSpan]]:
        sentences = self.split_into_sentences(post_text)
        adaptive_spans = []
        maladaptive_spans = []
        
        context = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            prompt = ContextPrompt.CONTEXT_PROMPT.format(
                context=context if context else "[Beginning of post]",
                sentence=sentence
            )
            response = self.llm.generate(prompt, max_new_tokens=10)
            
            label = self._parse_classification_response(response)
            
            if label == "adaptive":
                adaptive_spans.append(SelfStateSpan(text=sentence, label="adaptive"))
            elif label == "maladaptive":
                maladaptive_spans.append(SelfStateSpan(text=sentence, label="maladaptive"))
            
            context += " " + sentence
        
        return adaptive_spans, maladaptive_spans
    
    def baseline_with_importance_classify(self, post_text: str) -> Tuple[List[SelfStateSpan], List[SelfStateSpan]]:
        sentences = self.split_into_sentences(post_text)
        adaptive_spans = []
        maladaptive_spans = []
        
        # First pass: filter important sentences
        important_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            importance_prompt = ImportancePrompt.IMPORTANCE_PROMPT.format(sentence=sentence)
            importance_response = self.llm.generate(importance_prompt, max_new_tokens=10)
            
            if "important" in importance_response.lower():
                important_sentences.append(sentence)
        
        # Second pass: classify important sentences with context
        context = ""
        for sentence in important_sentences:
            prompt = ContextPrompt.CONTEXT_PROMPT.format(
                context=context if context else "[Beginning of post]",
                sentence=sentence
            )
            response = self.llm.generate(prompt, max_new_tokens=10)
            
            label = self._parse_classification_response(response)
            
            if label == "adaptive":
                adaptive_spans.append(SelfStateSpan(text=sentence, label="adaptive"))
            elif label == "maladaptive":
                maladaptive_spans.append(SelfStateSpan(text=sentence, label="maladaptive"))
            
            context += " " + sentence
        
        return adaptive_spans, maladaptive_spans
    
    def span_identification_classify(self, post_text: str, adaptive_boost: bool = False) -> Tuple[List[SelfStateSpan], List[SelfStateSpan]]:
        sentences = self.split_into_sentences(post_text)
        adaptive_spans = []
        maladaptive_spans = []
        
        # Create 2-sentence chunks
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i:i+2])
            if not chunk.strip():
                continue
            
            if adaptive_boost:
                prompt = SpanIdentificationAdaptiveBoostPrompt.SPAN_IDENTIFICATION_ADAPTIVE_BOOST_PROMPT.format(chunk=chunk)
            else:
                prompt = SpanIdentificationPrompt.SPAN_IDENTIFICATION_PROMPT.format(chunk=chunk)
            
            response = self.llm.generate(prompt, max_new_tokens=500)
            
            spans = self._parse_span_response(response)
            
            for span in spans:
                if span.label == "adaptive":
                    adaptive_spans.append(span)
                elif span.label == "maladaptive":
                    maladaptive_spans.append(span)
        
        return adaptive_spans, maladaptive_spans
    
    def _parse_classification_response(self, response: str) -> str:
        response_lower = response.lower().strip()
        
        # Extract first word
        first_word = response_lower.split()[0] if response_lower else ""
        
        if "adaptive" in first_word and "maladaptive" not in first_word:
            return "adaptive"
        elif "maladaptive" in first_word:
            return "maladaptive"
        elif "neither" in first_word:
            return "neither"
        
        # Fallback: check entire response
        if "maladaptive" in response_lower:
            return "maladaptive"
        elif "adaptive" in response_lower:
            return "adaptive"
        
        return "neither"
    
    def _parse_span_response(self, response: str) -> List[SelfStateSpan]:
        spans = []
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                for item in data:
                    if isinstance(item, dict) and 'phrase' in item and 'label' in item:
                        spans.append(SelfStateSpan(
                            text=item['phrase'],
                            label=item['label'].lower()
                        ))
        except json.JSONDecodeError:
            pass
        
        return spans