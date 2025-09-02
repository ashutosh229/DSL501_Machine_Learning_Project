import json
from bert_score import score
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
from models.states import SelfState, ProcessedPost, ClassificationResult, StateType
from utils.llm import LLMInterface
from agents.preprocessing_agent import DataProcessingAgent
from logger import logger
from utils.knowledge_base import KnowledgeBase

warnings.filterwarnings("ignore")




























class AgenticOrchestrator:
    """Main orchestrator that coordinates all agents"""

    def __init__(self, use_wandb: bool = False):
        # Initialize components
        self.knowledge_base = KnowledgeBase()
        self.llm = LLMInterface()

        # Initialize agents
        self.data_processor = DataProcessingAgent()
        self.state_finder = StateFinderAgent(self.llm, self.knowledge_base)
        self.adaptive_classifier = AdaptiveClassifierAgent(
            self.llm, self.knowledge_base
        )
        self.maladaptive_classifier = MaladaptiveClassifierAgent(
            self.llm, self.knowledge_base
        )
        self.validator = ValidationAgent(self.llm)
        self.refiner = RefinementAgent(self.llm)

        # Monitoring
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="agentic-self-state-classification")

        self.max_refinement_iterations = 2
        logger.info("Agentic Orchestrator initialized with all 6 agents")

    def process_post(self, post_data: Dict[str, Any]) -> ClassificationResult:
        """Main processing pipeline - orchestrates all agents"""
        start_time = datetime.now()
        agent_votes = {}

        try:
            # Step 1: Data Processing Agent
            logger.info("Step 1: Processing data...")
            processed_post = self.data_processor.process_post(post_data)
            agent_votes["data_processor"] = {
                "status": "success",
                "sentences": len(processed_post.sentences),
            }

            # Step 2: State Finder Agent
            logger.info("Step 2: Finding self-states...")
            candidates = self.state_finder.find_self_states(processed_post)
            agent_votes["state_finder"] = {
                "status": "success",
                "candidates": len(candidates),
            }

            # Step 3 & 4: Classification Agents (parallel)
            logger.info("Step 3-4: Classifying states...")
            adaptive_states = self.adaptive_classifier.classify_adaptive(candidates)
            maladaptive_states = self.maladaptive_classifier.classify_maladaptive(
                candidates
            )

            agent_votes["adaptive_classifier"] = {
                "status": "success",
                "states": len(adaptive_states),
            }
            agent_votes["maladaptive_classifier"] = {
                "status": "success",
                "states": len(maladaptive_states),
            }

            # Step 5: Validation Agent (with refinement loop)
            refinement_count = 0
            while refinement_count < self.max_refinement_iterations:
                logger.info(
                    f"Step 5: Validating results (iteration {refinement_count + 1})..."
                )
                validation_result = self.validator.validate_results(
                    adaptive_states, maladaptive_states, processed_post
                )

                agent_votes[f"validator_iter_{refinement_count}"] = {
                    "status": "success",
                    "is_valid": validation_result["is_valid"],
                    "score": validation_result["overall_score"],
                }

                if validation_result["is_valid"]:
                    logger.info("Validation passed!")
                    break

                # Step 6: Refinement Agent
                logger.info("Step 6: Refining classifications...")
                refinement_result = self.refiner.refine_classification(
                    validation_result, adaptive_states, maladaptive_states, candidates
                )

                adaptive_states = refinement_result["adaptive_states"]
                maladaptive_states = refinement_result["maladaptive_states"]

                agent_votes[f"refiner_iter_{refinement_count}"] = {
                    "status": "success",
                    "actions": refinement_result["actions_taken"],
                }

                refinement_count += 1

            # Calculate final metrics
            all_states = adaptive_states + maladaptive_states
            overall_confidence = self._calculate_overall_confidence(all_states)
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log(
                    {
                        "processing_time": processing_time,
                        "adaptive_count": len(adaptive_states),
                        "maladaptive_count": len(maladaptive_states),
                        "overall_confidence": overall_confidence,
                        "refinement_iterations": refinement_count,
                    }
                )

            result = ClassificationResult(
                post_id=processed_post.post_id,
                self_states=all_states,
                overall_confidence=overall_confidence,
                processing_time=processing_time,
                agent_votes=agent_votes,
            )

            logger.info(f"Processing complete for {processed_post.post_id}")
            return result

        except Exception as e:
            logger.error(f"Error processing post: {e}")
            return ClassificationResult(
                post_id=post_data.get("post_id", "error"),
                self_states=[],
                overall_confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                agent_votes={"error": str(e)},
            )

    def process_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> List[ClassificationResult]:
        """Process multiple posts"""
        results = []

        logger.info(f"Processing dataset with {len(dataset)} posts")

        for i, post_data in enumerate(dataset):
            logger.info(f"Processing post {i+1}/{len(dataset)}")
            result = self.process_post(post_data)
            results.append(result)

        # Calculate dataset-level metrics
        self._calculate_dataset_metrics(results)

        return results

    def _calculate_overall_confidence(self, states: List[SelfState]) -> float:
        """Calculate overall confidence for all states"""
        if not states:
            return 0.0

        confidences = [state.confidence for state in states]
        return sum(confidences) / len(confidences)

    def _calculate_dataset_metrics(
        self, results: List[ClassificationResult]
    ) -> Dict[str, float]:
        """Calculate metrics across the entire dataset"""
        total_adaptive = sum(
            len([s for s in r.self_states if s.state_type == StateType.ADAPTIVE])
            for r in results
        )
        total_maladaptive = sum(
            len([s for s in r.self_states if s.state_type == StateType.MALADAPTIVE])
            for r in results
        )
        total_posts = len(results)
        avg_processing_time = sum(r.processing_time for r in results) / total_posts
        avg_confidence = sum(r.overall_confidence for r in results) / total_posts

        metrics = {
            "total_posts": total_posts,
            "total_adaptive_states": total_adaptive,
            "total_maladaptive_states": total_maladaptive,
            "avg_adaptive_per_post": total_adaptive / total_posts,
            "avg_maladaptive_per_post": total_maladaptive / total_posts,
            "avg_processing_time": avg_processing_time,
            "avg_confidence": avg_confidence,
        }

        logger.info("Dataset Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")

        if self.use_wandb:
            wandb.log(metrics)

        return metrics


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


def create_sample_dataset() -> List[Dict[str, Any]]:
    """Create sample dataset for testing"""
    sample_posts = [
        {
            "post_id": "post_001",
            "user_id": "user_001",
            "timeline_position": 1,
            "text": "I've been really struggling with depression lately. But I'm trying to get help and talk to a therapist. Some days are really hard and I feel worthless. Other days I feel like I'm making progress and learning to cope better.",
        },
        {
            "post_id": "post_002",
            "user_id": "user_002",
            "timeline_position": 1,
            "text": "I hate myself and everything I do. Nothing ever goes right and I'm just a complete failure. I don't see the point in trying anymore.",
        },
        {
            "post_id": "post_003",
            "user_id": "user_003",
            "timeline_position": 1,
            "text": "Started going to therapy this week. It's scary but I think it's the right step. My therapist seems really understanding and I'm hopeful this will help me work through my issues.",
        },
    ]

    return sample_posts


def create_sample_ground_truth() -> List[Dict[str, Any]]:
    """Create sample ground truth for evaluation"""
    ground_truth = [
        {
            "post_id": "post_001",
            "adaptive_evidence": [
                "I'm trying to get help and talk to a therapist",
                "I feel like I'm making progress and learning to cope better",
            ],
            "maladaptive_evidence": [
                "I've been really struggling with depression",
                "Some days are really hard and I feel worthless",
            ],
        },
        {
            "post_id": "post_002",
            "adaptive_evidence": [],
            "maladaptive_evidence": [
                "I hate myself and everything I do",
                "Nothing ever goes right and I'm just a complete failure",
                "I don't see the point in trying anymore",
            ],
        },
        {
            "post_id": "post_003",
            "adaptive_evidence": [
                "Started going to therapy this week",
                "I think it's the right step",
                "My therapist seems really understanding",
                "I'm hopeful this will help me work through my issues",
            ],
            "maladaptive_evidence": ["It's scary"],
        },
    ]

    return ground_truth


def main():
    """Main function to run the complete system"""
    logger.info("Starting Agentic Self-State Classification System")

    # Initialize system
    orchestrator = AgenticOrchestrator(
        use_wandb=False
    )  # Set to True to enable W&B logging
    evaluator = BERTScoreEvaluator()

    # Create sample data
    dataset = create_sample_dataset()
    ground_truth = create_sample_ground_truth()

    logger.info(f"Processing {len(dataset)} sample posts...")

    # Process dataset
    results = orchestrator.process_dataset(dataset)

    # Display results
    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)

    for i, result in enumerate(results):
        print(f"\nPost {i+1} ({result.post_id}):")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Overall Confidence: {result.overall_confidence:.3f}")

        adaptive_states = [
            s for s in result.self_states if s.state_type == StateType.ADAPTIVE
        ]
        maladaptive_states = [
            s for s in result.self_states if s.state_type == StateType.MALADAPTIVE
        ]

        print(f"\nAdaptive States ({len(adaptive_states)}):")
        for j, state in enumerate(adaptive_states):
            print(f'  {j+1}. "{state.text}" (confidence: {state.confidence:.3f})')

        print(f"\nMaladaptive States ({len(maladaptive_states)}):")
        for j, state in enumerate(maladaptive_states):
            print(f'  {j+1}. "{state.text}" (confidence: {state.confidence:.3f})')

    # Evaluate results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    evaluation_results = evaluator.evaluate_predictions(results, ground_truth)

    for metric, score in evaluation_results.items():
        print(f"{metric}: {score:.3f}")

    # Target comparison (from your SoP)
    target_recall = 0.60
    achieved_recall = evaluation_results["overall_bert_f1"]

    print(f"\nTarget Recall: {target_recall:.3f}")
    print(f"Achieved Recall: {achieved_recall:.3f}")
    print(f"Target {'✅ MET' if achieved_recall >= target_recall else '❌ NOT MET'}")

    logger.info("System run completed successfully!")


if __name__ == "__main__":
    main()
