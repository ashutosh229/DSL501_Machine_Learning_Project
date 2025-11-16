class BaselinePrompt:  
    BASELINE_PROMPT = """You are a professional psychologist. Given a social media post, classify whether or not a sentence demonstrates an adaptive or maladaptive self-state.

An adaptive self-state reflects aspects of the self that are flexible, non-ruminative, and promote well-being and optimal functioning.

A maladaptive self-state reflects internal states or perspectives that hinder an individual's ability to adapt to situations or cope with challenges effectively, potentially leading to emotional distress or behavioral problems.

Here is the sentence: {sentence}

Respond with ONLY one word: "adaptive", "maladaptive", or "neither"."""