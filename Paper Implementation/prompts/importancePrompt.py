class ImportancePrompt:
    IMPORTANCE_PROMPT = """You are a professional psychologist analyzing a sentence from a social media post.

    Determine if this sentence contains ANY reference to the following MIND self-state dimensions:
    1. Affective (emotional expressions)
    2. Behavior-self (actions toward oneself)
    3. Behavior-others (actions toward others)
    4. Cognition-self (thoughts about oneself)
    5. Cognition-others (thoughts about others)
    6. Desire (needs, goals, intentions, fears)

    Sentence: {sentence}

    Respond with ONLY one word: "important" or "unimportant"."""