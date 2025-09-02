from utils.knowledge_base import KnowledgeBase
from utils.llm import LLMInterface
from agents.preprocessing_agent import DataProcessingAgent
from agents.state_finder_agent import StateFinderAgent
from agents.adaptive_classifier_agent import AdaptiveClassifierAgent
from agents.maladaptive_classifier_agent import MaladaptiveClassifierAgent
from agents.validation_agent import ValidationAgent
from agents.refinement_agent import RefinementAgent
from logger import logger
import wandb
from models.states import ClassificationResult
from typing import Dict, Any, List
from datetime import datetime
from models.states import StateType, SelfState


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
