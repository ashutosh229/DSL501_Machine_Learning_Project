class LLMInterface:
    """Interface for Gemma 2 9B model with 4-bit quantization"""

    def __init__(self):
        self.model_name = "google/gemma-2-9b-it"
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load 4-bit quantized Gemma 2 9B model"""
        try:
            logger.info("Loading Gemma 2 9B model with 4-bit quantization...")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to CPU-only mode or mock responses for testing
            self.model = None
            self.tokenizer = None

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from the model"""
        if self.model is None:
            # Mock response for testing when model isn't available
            return "Mock response - model not loaded"

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response.replace(prompt, "").strip()
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"
