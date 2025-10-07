class MockLLM:

    def __init__(self):
        print("Using Mock LLM (for testing without actual model)")
        
    def load_model(self):
        pass
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.1) -> str:
        prompt_lower = prompt.lower()
        
        # For importance filtering
        if "important" in prompt_lower and "unimportant" in prompt_lower:
            psych_keywords = ['feel', 'think', 'want', 'need', 'sad', 'happy', 'angry', 
                            'depressed', 'anxious', 'hope', 'fear', 'love', 'hate']
            if any(kw in prompt_lower for kw in psych_keywords):
                return "important"
            return "unimportant"
        
        # For classification
        if "adaptive" in prompt_lower and "maladaptive" in prompt_lower:
            negative_keywords = ['suicide', 'kill', 'worthless', 'hate myself', 'hopeless', 
                               'give up', 'can\'t do', 'failure', 'useless', 'alone']
            positive_keywords = ['therapy', 'help', 'support', 'friend', 'plan', 'try',
                               'getting better', 'learned', 'grow', 'hope']
            
            has_negative = any(kw in prompt_lower for kw in negative_keywords)
            has_positive = any(kw in prompt_lower for kw in positive_keywords)
            
            if has_negative and not has_positive:
                return "maladaptive"
            elif has_positive:
                return "adaptive"
            return "neither"
        
        # For span identification
        if "json" in prompt_lower and "phrase" in prompt_lower:
            return "[]"
        
        return "neither"
