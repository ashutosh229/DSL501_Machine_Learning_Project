class ContextPrompt:  
    CONTEXT_PROMPT = """You are a professional psychologist. Given a social media post, classify whether or not a sentence demonstrates an adaptive or maladaptive self-state.

An adaptive self-state reflects internal processes that are flexible, constructive, and promote emotional well-being, effective functioning, and psychological health.

A maladaptive self-state reflects internal processes that are rigid, ruminative, self-defeating, or harmful, and are often associated with emotional distress or impaired functioning.

To make your classification, use the ABCD framework for psychological self-states:

A. **Affect** -- Type of emotional expression
- Adaptive: calm, content, assertive, proud, justifiable pain/grief
- Maladaptive: anxious, hopeless, apathetic, aggressive, ashamed, depressed

B. **Behavior** -- Main behavioral tendencies
- Toward Others (BO):
  - Adaptive: relational, autonomous behavior
  - Maladaptive: fight/flight response, controlling or overcontrolled behavior
- Toward Self (BS):
  - Adaptive: self-care
  - Maladaptive: self-neglect, avoidance, self-harm

C. **Cognition** -- Main thought patterns
- Toward Others (CO):
  - Adaptive: perceiving others as supportive or related
  - Maladaptive: perceiving others as detached, overattached, or autonomy-blocking
- Toward Self (CS):
  - Adaptive: self-compassion and acceptance
  - Maladaptive: self-criticism

D. **Desire** -- Expressed needs, goals, intentions, or fears
- Adaptive: desire for autonomy, relatedness, self-esteem, care
- Maladaptive: fear that these needs won't be met

Here are a couple of examples:

"I feel completely numb and don't care about anything anymore"
This is maladaptive. It shows a bluntedness and apathetic affective state.

"I broke down crying when talking to my therapist, which felt like a release"
This is adaptive. The crying is not a sign of maladaptive self-state, rather it is a healthy sadness.

You will be shown:
1. The context of the post so far
2. The current sentence to classify

If the sentence clearly demonstrates one or more **maladaptive or adaptive self-state(s)** based on this framework, classify it accordingly.

Here is the post so far:
{context}

Here is the current sentence:
{sentence}

Respond with ONLY one word: "adaptive", "maladaptive", or "neither"."""