from logger import logger
from utils.llm import LLMInterface
from utils.knowledge_base import KnowledgeBase
from typing import List, Dict, Any
from models.states import SelfState, StateType


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
            for span in candidate["potential_spans"]:
                if self._is_maladaptive_state(span, candidate["text"]):
                    confidence = self._calculate_maladaptive_confidence(span)
                    reasoning = self._generate_maladaptive_reasoning(span)

                    if confidence >= self.sensitivity:
                        maladaptive_states.append(
                            SelfState(
                                text=span,
                                state_type=StateType.MALADAPTIVE,
                                confidence=confidence,
                                reasoning=reasoning,
                                span_start=candidate["text"].find(span),
                                span_end=candidate["text"].find(span) + len(span),
                            )
                        )

        logger.info(f"Classified {len(maladaptive_states)} maladaptive states")
        return maladaptive_states

    def _is_maladaptive_state(self, span: str, context: str) -> bool:
        """Determine if span represents maladaptive state"""
        maladaptive_indicators = [
            "hate",
            "worthless",
            "hopeless",
            "useless",
            "failure",
            "hurt",
            "pain",
            "suffer",
            "destroy",
            "ruin",
            "never",
            "always",
            "terrible",
            "awful",
            "horrible",
        ]

        return any(indicator in span.lower() for indicator in maladaptive_indicators)

    def _calculate_maladaptive_confidence(self, span: str) -> float:
        """Calculate confidence for maladaptive classification"""
        patterns = self.kb.get_relevant_patterns(span, StateType.MALADAPTIVE)

        base_confidence = 0.6
        pattern_boost = len(patterns) * 0.15

        # Boost for explicit negative terms
        explicit_terms = ["hate myself", "want to die", "worthless", "hopeless"]
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
