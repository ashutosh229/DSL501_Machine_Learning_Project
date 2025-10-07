from scripts.selfstateClassifier import SelfStateClassifier
from scripts.llm.real import LLMInterface
from scripts.experimentRunner import ExperimentRunner

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
        llm = LLMInterface(model_name="google/gemma-2-9b-it", use_4bit=True)
        classifier = SelfStateClassifier(llm_interface=llm)
    
    # Create sample test data (since we don't have actual CLPsych data)
    print("\nCreating sample test data...")
    sample_posts = [
        Post(
            post_id="post_1",
            text="I've been feeling really hopeless lately. Everything seems pointless. But I did reach out to my therapist and we made a plan for next week. I'm going to try some new coping strategies.",
            adaptive_evidence=["I did reach out to my therapist and we made a plan for next week", "I'm going to try some new coping strategies"],
            maladaptive_evidence=["I've been feeling really hopeless lately", "Everything seems pointless"]
        ),
        Post(
            post_id="post_2",
            text="I can't do anything right. I'm such a failure. My friends tried to help me but I just pushed them away. I hate myself for that.",
            adaptive_evidence=[],
            maladaptive_evidence=["I can't do anything right", "I'm such a failure", "I hate myself for that"]
        ),
        Post(
            post_id="post_3",
            text="Today was tough but I made it through. I went for a walk which helped clear my head. I'm learning to be more patient with myself.",
            adaptive_evidence=["I went for a walk which helped clear my head", "I'm learning to be more patient with myself"],
            maladaptive_evidence=[]
        )
    ]
    
    print(f"Created {len(sample_posts)} sample posts")
    
    # Run experiments
    print("\n" + "="*60)
    print("Running Experiments")
    print("="*60)
    
    runner = ExperimentRunner(classifier)
    
    # Test individual methods
    print("\nTesting Baseline method...")
    baseline_results = runner.run_method("baseline", sample_posts)
    
    print("\nTesting Baseline + Context method...")
    context_results = runner.run_method("baseline_context", sample_posts)
    
    # Compare all methods
    print("\n" + "="*60)
    print("Comparing All Methods")
    print("="*60)
    
    all_results = runner.compare_methods(sample_posts)
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
    
    print("\n" + "="*60)
    print("Setup Instructions for Real Data:")
    print("="*60)