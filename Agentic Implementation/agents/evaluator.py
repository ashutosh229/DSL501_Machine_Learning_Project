from logger import logger
from typing import List, Dict, Any
from bert_score import score
from models.states import ClassificationResult, SelfState, StateType


class BERTScoreEvaluator:
    """Evaluation using BERTScore as specified in CLPsych 2025"""

    def __init__(self):
        try:
            self.bert_score_fn = score
            logger.info("BERTScore evaluator initialized")
        except ImportError:
            logger.warning(
                "bert-score not installed. Install with: pip install bert-score"
            )
            self.bert_score_fn = None

    def evaluate_predictions(
        self,
        predictions: List[ClassificationResult],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth using BERTScore"""

        if self.bert_score_fn is None:
            return self._mock_evaluation(predictions, ground_truth)

        adaptive_predictions = []
        adaptive_ground_truth = []
        maladaptive_predictions = []
        maladaptive_ground_truth = []

        # Separate adaptive and maladaptive for evaluation
        for pred, gt in zip(predictions, ground_truth):
            # Extract adaptive states
            adaptive_pred_texts = [
                s.text for s in pred.self_states if s.state_type == StateType.ADAPTIVE
            ]
            adaptive_gt_texts = gt.get("adaptive_evidence", [])

            adaptive_predictions.extend(adaptive_pred_texts)
            adaptive_ground_truth.extend(adaptive_gt_texts)

            # Extract maladaptive states
            maladaptive_pred_texts = [
                s.text
                for s in pred.self_states
                if s.state_type == StateType.MALADAPTIVE
            ]
            maladaptive_gt_texts = gt.get("maladaptive_evidence", [])

            maladaptive_predictions.extend(maladaptive_pred_texts)
            maladaptive_ground_truth.extend(maladaptive_gt_texts)

        # Calculate BERTScore
        results = {}

        if adaptive_predictions and adaptive_ground_truth:
            _, _, adaptive_f1 = self.bert_score_fn(
                adaptive_predictions, adaptive_ground_truth, lang="en"
            )
            results["adaptive_bert_f1"] = float(adaptive_f1.mean())
        else:
            results["adaptive_bert_f1"] = 0.0

        if maladaptive_predictions and maladaptive_ground_truth:
            _, _, maladaptive_f1 = self.bert_score_fn(
                maladaptive_predictions, maladaptive_ground_truth, lang="en"
            )
            results["maladaptive_bert_f1"] = float(maladaptive_f1.mean())
        else:
            results["maladaptive_bert_f1"] = 0.0

        # Overall score
        results["overall_bert_f1"] = (
            results["adaptive_bert_f1"] + results["maladaptive_bert_f1"]
        ) / 2

        logger.info("BERTScore Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.3f}")

        return results

    def _mock_evaluation(
        self,
        predictions: List[ClassificationResult],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Mock evaluation when BERTScore is not available"""
        logger.info("Using mock evaluation (BERTScore not available)")

        total_predictions = sum(len(p.self_states) for p in predictions)
        total_ground_truth = sum(
            len(gt.get("adaptive_evidence", []))
            + len(gt.get("maladaptive_evidence", []))
            for gt in ground_truth
        )

        # Simple overlap-based mock score
        mock_recall = min(1.0, total_predictions / max(1, total_ground_truth))

        return {
            "adaptive_bert_f1": mock_recall * 0.8,  # Assume lower for adaptive
            "maladaptive_bert_f1": mock_recall * 0.9,  # Assume higher for maladaptive
            "overall_bert_f1": mock_recall * 0.85,
        }
