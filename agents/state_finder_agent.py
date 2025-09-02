from logger import logger
from utils.llm import LLMInterface
from utils.knowledge_base import KnowledgeBase
from models.states import ProcessedPost
from typing import List, Dict, Any


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

            if observation["has_self_state"]:
                candidates.append(
                    {
                        "text": window,
                        "window_index": i,
                        "potential_spans": observation["spans"],
                        "reasoning": f"Thought: {thought}\nAction: {action}\nObservation: {observation}",
                    }
                )

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
        has_self_state = any(
            keyword in window.lower()
            for keyword in [
                "feel",
                "think",
                "want",
                "need",
                "hope",
                "fear",
                "i am",
                "i was",
            ]
        )

        spans = []
        if has_self_state:
            # Extract sentences that likely contain self-states
            sentences = window.split(".")
            for sentence in sentences:
                if any(
                    keyword in sentence.lower()
                    for keyword in ["feel", "think", "want", "need", "hope", "fear"]
                ):
                    spans.append(sentence.strip())

        return {
            "has_self_state": has_self_state,
            "spans": spans,
            "confidence": 0.7 if has_self_state else 0.1,
        }
