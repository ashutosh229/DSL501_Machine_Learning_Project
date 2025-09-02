from logger import logger
from utils.llm import LLMInterface
from utils.knowledge_base import KnowledgeBase
from typing import List, Dict, Any
from models.states import SelfState, StateType


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
            for span in candidate["potential_spans"]:
                if self._is_adaptive_state(span, candidate["text"]):
                    confidence = self._calculate_adaptive_confidence(span)
                    reasoning = self._generate_adaptive_reasoning(span)

                    if confidence >= self.sensitivity:
                        adaptive_states.append(
                            SelfState(
                                text=span,
                                state_type=StateType.ADAPTIVE,
                                confidence=confidence,
                                reasoning=reasoning,
                                span_start=candidate["text"].find(span),
                                span_end=candidate["text"].find(span) + len(span),
                            )
                        )

        logger.info(f"Classified {len(adaptive_states)} adaptive states")
        return adaptive_states

    def _is_adaptive_state(self, span: str, context: str) -> bool:
        """Determine if span represents adaptive state"""
        adaptive_indicators = [
            "help",
            "support",
            "better",
            "improve",
            "plan",
            "goal",
            "grateful",
            "thankful",
            "progress",
            "learn",
            "grow",
            "cope",
            "manage",
            "handle",
            "overcome",
            "resilient",
        ]

        return any(indicator in span.lower() for indicator in adaptive_indicators)

    def _calculate_adaptive_confidence(self, span: str) -> float:
        """Calculate confidence for adaptive classification"""
        # Get relevant patterns from knowledge base
        patterns = self.kb.get_relevant_patterns(span, StateType.ADAPTIVE)

        base_confidence = 0.5
        pattern_boost = len(patterns) * 0.1

        # Boost confidence for subtle adaptive indicators
        subtle_indicators = ["trying", "working on", "hoping", "planning"]
        subtle_boost = sum(
            0.05 for indicator in subtle_indicators if indicator in span.lower()
        )

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
