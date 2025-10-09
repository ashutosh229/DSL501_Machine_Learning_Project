import warnings
from logger import logger
from agents.orchestrator import AgenticOrchestrator
from agents.evaluator import BERTScoreEvaluator
from utils.dataset import create_sample_dataset, create_sample_ground_truth
from models.stateType import StateType

warnings.filterwarnings("ignore")


def main():
    logger.info("Starting Agentic Self-State Classification System")
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
