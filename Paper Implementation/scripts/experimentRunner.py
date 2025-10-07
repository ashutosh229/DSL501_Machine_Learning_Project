from typing import List, Dict
from collections import defaultdict
from models.redditPost import Post
from scripts.selfStateClassifier import SelfStateClassifier
from scripts.evaluator import Evaluator


class ExperimentRunner:

    def __init__(self, classifier: SelfStateClassifier):
        self.classifier = classifier
        self.results = defaultdict(list)
    
    def run_method(self, method_name: str, posts: List[Post]) -> Dict[str, float]:
        all_adaptive_preds = []
        all_maladaptive_preds = []
        all_adaptive_gold = []
        all_maladaptive_gold = []
        
        for post in posts:
            # Run classification
            if method_name == "baseline":
                adaptive_spans, maladaptive_spans = self.classifier.baseline_classify(post.text)
            elif method_name == "baseline_context":
                adaptive_spans, maladaptive_spans = self.classifier.baseline_with_context_classify(post.text)
            elif method_name == "baseline_importance":
                adaptive_spans, maladaptive_spans = self.classifier.baseline_with_importance_classify(post.text)
            elif method_name == "span_id":
                adaptive_spans, maladaptive_spans = self.classifier.span_identification_classify(post.text, adaptive_boost=False)
            elif method_name == "span_id_boost":
                adaptive_spans, maladaptive_spans = self.classifier.span_identification_classify(post.text, adaptive_boost=True)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Collect predictions and gold annotations
            all_adaptive_preds.extend([span.text for span in adaptive_spans])
            all_maladaptive_preds.extend([span.text for span in maladaptive_spans])
            all_adaptive_gold.extend(post.adaptive_evidence)
            all_maladaptive_gold.extend(post.maladaptive_evidence)
        
        # Evaluate
        results = Evaluator.evaluate_predictions(
            all_adaptive_preds,
            all_maladaptive_preds,
            all_adaptive_gold,
            all_maladaptive_gold
        )
        
        return results
    
    def compare_methods(self, posts: List[Post], methods: List[str] = None) -> Dict[str, Dict[str, float]]:
        if methods is None:
            methods = ["baseline", "baseline_context", "baseline_importance", "span_id", "span_id_boost"]
        
        results = {}
        for method in methods:
            print(f"\nRunning method: {method}")
            results[method] = self.run_method(method, posts)
            print(f"Results: Overall Recall = {results[method]['overall_recall']:.3f}")
        
        return results
    
    @staticmethod
    def print_results_table(results: Dict[str, Dict[str, float]]):
        print("\n" + "="*100)
        print("RESULTS COMPARISON")
        print("="*100)
        
        headers = ["Method", "Overall Recall", "Recall (A)", "Recall (M)", 
                  "Weighted Recall", "Weighted (A)", "Weighted (M)"]
        
        print(f"{headers[0]:<25} {headers[1]:<15} {headers[2]:<12} {headers[3]:<12} "
              f"{headers[4]:<17} {headers[5]:<14} {headers[6]:<14}")
        print("-"*100)
        
        for method, scores in results.items():
            print(f"{method:<25} "
                  f"{scores['overall_recall']:<15.3f} "
                  f"{scores['adaptive_recall']:<12.3f} "
                  f"{scores['maladaptive_recall']:<12.3f} "
                  f"{scores['overall_weighted_recall']:<17.3f} "
                  f"{scores['adaptive_weighted_recall']:<14.3f} "
                  f"{scores['maladaptive_weighted_recall']:<14.3f}")
        
        print("="*100)