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
