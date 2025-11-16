class SpanIdentificationAdaptiveBoostPrompt:
    SPAN_IDENTIFICATION_ADAPTIVE_BOOST_PROMPT = """You are a professional psychologist. Analyze the following text chunk and identify specific phrases that demonstrate adaptive or maladaptive self-states.

    IMPORTANT: Pay careful attention to SUBTLE adaptive self-states that may be hidden within seemingly negative sentences. Adaptive self-states can include:
    - Asking for help
    - Making plans or taking action
    - Expressing emotions appropriately (even negative emotions like appropriate sadness or anger)
    - Showing self-awareness
    - Any behavior indicating coping or resilience

    An adaptive self-state reflects internal processes that are flexible, constructive, and promote emotional well-being.
    A maladaptive self-state reflects internal processes that are rigid, ruminative, self-defeating, or harmful.

    Text chunk:
    {chunk}

    Identify ALL phrases (including very subtle ones) that demonstrate self-states. Try to annotate as much of the text as possible. For each phrase, provide:
    1. The exact phrase text
    2. Whether it's "adaptive" or "maladaptive"

    Format your response as a JSON list of objects like this:
    [
    {{"phrase": "exact text from chunk", "label": "adaptive"}},
    {{"phrase": "another exact text", "label": "maladaptive"}}
    ]

    If no self-states are found, return an empty list: []"""