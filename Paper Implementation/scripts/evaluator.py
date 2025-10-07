from typing import List, Dict
import numpy as np

class Evaluator:

    @staticmethod
    def simple_token_overlap(pred: str, gold: str) -> float:
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        intersection = pred_tokens.intersection(gold_tokens)
        union = pred_tokens.union(gold_tokens)
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def calculate_recall(predictions: List[str], gold_annotations: List[str]) -> float:
        if not gold_annotations:
            return 0.0
        
        max_scores = []
        for gold in gold_annotations:
            if not gold.strip():
                continue
            
            scores = [Evaluator.simple_token_overlap(pred, gold) for pred in predictions]
            max_score = max(scores) if scores else 0.0
            max_scores.append(max_score)
        
        return np.mean(max_scores) if max_scores else 0.0
    
    @staticmethod
    def calculate_weighted_recall(predictions: List[str], gold_annotations: List[str]) -> float:
        pred_token_count = sum(len(pred.split()) for pred in predictions)
        gold_token_count = sum(len(gold.split()) for gold in gold_annotations)
        
        if gold_token_count == 0:
            return 0.0
        
        base_recall = Evaluator.calculate_recall(predictions, gold_annotations)
        
        # Weight by how close prediction count is to gold count
        count_ratio = min(pred_token_count, gold_token_count) / max(pred_token_count, gold_token_count)
        
        return base_recall * count_ratio
    
    @staticmethod
    def evaluate_predictions(
        adaptive_preds: List[str],
        maladaptive_preds: List[str],
        adaptive_gold: List[str],
        maladaptive_gold: List[str]
    ) -> Dict[str, float]:
        results = {
            'adaptive_recall': Evaluator.calculate_recall(adaptive_preds, adaptive_gold),
            'maladaptive_recall': Evaluator.calculate_recall(maladaptive_preds, maladaptive_gold),
            'adaptive_weighted_recall': Evaluator.calculate_weighted_recall(adaptive_preds, adaptive_gold),
            'maladaptive_weighted_recall': Evaluator.calculate_weighted_recall(maladaptive_preds, maladaptive_gold),
        }
        
        results['overall_recall'] = (results['adaptive_recall'] + results['maladaptive_recall']) / 2
        results['overall_weighted_recall'] = (
            results['adaptive_weighted_recall'] + results['maladaptive_weighted_recall']
        ) / 2
        
        return results