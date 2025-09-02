from logger import logger
from typing import Dict, List, Any, Tuple
from utils.llm import LLMInterface
from models.states import SelfState, StateType


class RefinementAgent:
    """Agent 6: Improves results when validation fails"""

    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.refinement_strategies = {
            "low_adaptive_recall": self._boost_adaptive_sensitivity,
            "high_false_positives": self._increase_thresholds,
            "inconsistent_results": self._resolve_conflicts,
            "low_confidence": self._improve_reasoning,
        }
        logger.info("Refinement Agent initialized")

    def refine_classification(
        self,
        validation_result: Dict[str, Any],
        adaptive_states: List[SelfState],
        maladaptive_states: List[SelfState],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Refine classification based on validation feedback"""

        issues = validation_result.get("flagged_issues", [])
        refinement_actions = []

        # Determine refinement strategy based on issues
        if "Many low-confidence adaptive classifications" in issues:
            refined_adaptive = self._boost_adaptive_sensitivity(
                adaptive_states, candidates
            )
            refinement_actions.append("boosted_adaptive_sensitivity")
        else:
            refined_adaptive = adaptive_states

        if (
            "Too many adaptive states detected" in issues
            or "Too many maladaptive states detected" in issues
        ):
            refined_adaptive, refined_maladaptive = self._increase_thresholds(
                refined_adaptive, maladaptive_states
            )
            refinement_actions.append("increased_thresholds")
        else:
            refined_maladaptive = maladaptive_states

        # Check for conflicts
        if validation_result.get("consistency_score", 1.0) < 0.7:
            refined_adaptive, refined_maladaptive = self._resolve_conflicts(
                refined_adaptive, refined_maladaptive
            )
            refinement_actions.append("resolved_conflicts")

        logger.info(f"Applied refinement actions: {refinement_actions}")

        return {
            "adaptive_states": refined_adaptive,
            "maladaptive_states": refined_maladaptive,
            "actions_taken": refinement_actions,
            "refinement_confidence": 0.8,
        }

    def _boost_adaptive_sensitivity(
        self, adaptive_states: List[SelfState], candidates: List[Dict[str, Any]]
    ) -> List[SelfState]:
        """Boost sensitivity for adaptive state detection"""
        boosted_states = list(adaptive_states)  # Keep existing states

        # Look for additional adaptive states in candidates with lower threshold
        for candidate in candidates:
            for span in candidate["potential_spans"]:
                # Look for subtle adaptive indicators
                subtle_adaptive = [
                    "trying",
                    "working",
                    "hoping",
                    "planning",
                    "learning",
                    "getting help",
                    "talking to",
                    "reaching out",
                ]

                if any(indicator in span.lower() for indicator in subtle_adaptive):
                    # Check if not already classified
                    if not any(span in state.text for state in boosted_states):
                        boosted_states.append(
                            SelfState(
                                text=span,
                                state_type=StateType.ADAPTIVE,
                                confidence=0.6,  # Lower threshold
                                reasoning="Detected through sensitivity boost - subtle adaptive indicator",
                                span_start=candidate["text"].find(span),
                                span_end=candidate["text"].find(span) + len(span),
                            )
                        )

        return boosted_states

    def _increase_thresholds(
        self, adaptive_states: List[SelfState], maladaptive_states: List[SelfState]
    ) -> Tuple[List[SelfState], List[SelfState]]:
        """Increase confidence thresholds to reduce false positives"""
        high_threshold = 0.8

        refined_adaptive = [
            s for s in adaptive_states if s.confidence >= high_threshold
        ]
        refined_maladaptive = [
            s for s in maladaptive_states if s.confidence >= high_threshold
        ]

        return refined_adaptive, refined_maladaptive

    def _resolve_conflicts(
        self, adaptive_states: List[SelfState], maladaptive_states: List[SelfState]
    ) -> Tuple[List[SelfState], List[SelfState]]:
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
                strong_indicators = ["definitely", "really", "very", "extremely"]
                if any(
                    indicator in state.text.lower() for indicator in strong_indicators
                ):
                    state.confidence = min(0.8, state.confidence + 0.2)

            improved_states.append(state)

        return improved_states
