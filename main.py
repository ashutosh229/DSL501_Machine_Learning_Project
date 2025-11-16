from scripts.selfStateClassifier import SelfStateClassifier
from scripts.llm.real import LLMInterface
from scripts.experimentRunner import ExperimentRunner
from scripts.dataLoader import DataLoader
import json

if __name__ == "__main__":
    print("Self-State Identification and Classification System")
    print("Based on CLPsych 2025 Task A.1")
    print("="*60)
    
    # Configuration
    USE_MOCK_LLM = False  # Set to False to use actual Gemma model
    
    # Initialize classifier
    if USE_MOCK_LLM:
        print("\nUsing Mock LLM for demonstration")
        classifier = SelfStateClassifier(use_mock=True)
    else:
        print("\nInitializing Gemma 2 9B with 4-bit quantization...")
        llm = LLMInterface(model_name="./resources/gemma-2-9b-it", use_4bit=True)
        classifier = SelfStateClassifier(llm_interface=llm)
    

    print("\nLoading the data...")
    timelines = DataLoader.load_all_timelines("data")
    all_posts = []
    for timeline in timelines:
        all_posts.extend(timeline.posts)

    # all_posts = all_posts[:10]

    
    print("Loaded the posts across all the timelines")
    
    # Run experiments
    print("\n" + "="*60)
    print("Running Experiments")
    print("="*60)
    
    runner = ExperimentRunner(classifier)
    
    # Test individual methods
    print("\nTesting Baseline method...")
    baseline_results = runner.run_method("baseline", all_posts)
    print(baseline_results)

    print("\nTesting Baseline + Context method...")
    context_results = runner.run_method("baseline_context", all_posts)
    print(context_results)

    # Compare all methods
    print("\n" + "="*60)
    print("Comparing All Methods")
    print("="*60)
    
    all_results = runner.compare_methods(all_posts)
    runner.print_results_table(all_results)
    
    # Save results
    print("\nSaving results to results.json...")
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print("\nKey Findings (as per paper):")
    print("1. Baseline + Context performs best overall")
    print("2. Importance filtering improves weighted recall but reduces overall recall")
    print("3. Span identification excels at maladaptive detection but struggles with adaptive")
    print("4. Adaptive boost helps balance adaptive/maladaptive recall")
    print("\nResults saved to: results.json")
    
    # Demo: Single post classification
    print("\n" + "="*60)
    print("Demo: Classifying a Single Post")
    print("="*60)
    
    demo_text = """I've been struggling a lot lately with depression. 
    Some days I can barely get out of bed. But today I managed to call 
    my therapist and schedule an appointment. Small steps, I guess."""
    
    print(f"\nPost text:\n{demo_text}")
    
    print("\n--- Baseline Method ---")
    adaptive, maladaptive = classifier.baseline_classify(demo_text)
    print(f"Adaptive spans ({len(adaptive)}):")
    for span in adaptive:
        print(f"  - {span.text}")
    print(f"Maladaptive spans ({len(maladaptive)}):")
    for span in maladaptive:
        print(f"  - {span.text}")
    
    print("\n--- Baseline + Context Method ---")
    adaptive, maladaptive = classifier.baseline_with_context_classify(demo_text)
    print(f"Adaptive spans ({len(adaptive)}):")
    for span in adaptive:
        print(f"  - {span.text}")
    print(f"Maladaptive spans ({len(maladaptive)}):")
    for span in maladaptive:
        print(f"  - {span.text}")