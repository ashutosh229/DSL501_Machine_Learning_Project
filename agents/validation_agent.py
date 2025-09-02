from logger import logger
from utils.llm import LLMInterface
from typing import List, Dict, Any
from models.states import SelfState, ProcessedPost, StateType


class ValidationAgent:
    """Agent 5: Validates and cross-checks classification results"""

    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.min_confidence = 0.6
        logger.info("Validation Agent initialized")

    def validate_results(
        self,
        adaptive_states: List[SelfState],
        maladaptive_states: List[SelfState],
        processed_post: ProcessedPost,
    ) -> Dict[str, Any]:
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
        overall_score = (
            consistency_score + confidence_distribution + temporal_consistency
        ) / 3

        validation_result = {
            "is_valid": overall_score >= self.min_confidence,
            "overall_score": overall_score,
            "consistency_score": consistency_score,
            "confidence_score": confidence_distribution,
            "temporal_score": temporal_consistency,
            "flagged_issues": self._identify_issues(
                adaptive_states, maladaptive_states
            ),
            "recommendation": (
                "accept" if overall_score >= self.min_confidence else "refine"
            ),
        }

        logger.info(
            f"Validation complete. Score: {overall_score:.2f}, Valid: {validation_result['is_valid']}"
        )
        return validation_result

    def _check_consistency(
        self, adaptive: List[SelfState], maladaptive: List[SelfState]
    ) -> float:
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

    def _check_temporal_consistency(
        self, all_states: List[SelfState], processed_post: ProcessedPost
    ) -> float:
        """Check temporal consistency (placeholder for timeline analysis)"""
        # For single post, check if states make sense together
        if len(all_states) <= 1:
            return 0.8

        # Simple heuristic: penalize if too many conflicting states
        adaptive_count = sum(
            1 for s in all_states if s.state_type == StateType.ADAPTIVE
        )
        maladaptive_count = len(all_states) - adaptive_count

        if adaptive_count > 0 and maladaptive_count > 0:
            # Mixed states are common in real data
            return 0.7
        else:
            # All one type might be less realistic
            return 0.6

    def _identify_issues(
        self, adaptive: List[SelfState], maladaptive: List[SelfState]
    ) -> List[str]:
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
