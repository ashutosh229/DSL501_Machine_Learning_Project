class SpanIdentificationPrompt:
    SPAN_IDENTIFICATION_PROMPT = """You are a professional psychologist. Analyze the following text chunk and identify specific phrases that demonstrate adaptive or maladaptive self-states.

    An adaptive self-state reflects internal processes that are flexible, constructive, and promote emotional well-being.
    A maladaptive self-state reflects internal processes that are rigid, ruminative, self-defeating, or harmful.

    Text chunk:
    {chunk}

    Identify ALL phrases (which may be sub-sentence level) that demonstrate self-states. For each phrase, provide:
    1. The exact phrase text
    2. Whether it's "adaptive" or "maladaptive"

    Format your response as a JSON list of objects like this:
    [
    {{"phrase": "exact text from chunk", "label": "adaptive"}},
    {{"phrase": "another exact text", "label": "maladaptive"}}
    ]

    If no self-states are found, return an empty list: []"""