from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import traceback

class LLMInterface:
    def __init__(self, model_name: str = "./resources/gemma-2-9b-it", use_4bit: bool = True, offload_folder: str = "./offload"):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.offload_folder = offload_folder
        self.model = None
        self.tokenizer = None

    def _gpu_total_free_gb(self):
        try:
            if not torch.cuda.is_available():
                return 0
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024 ** 3)
        except Exception:
            return 0

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

            quantization_config = None
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

            gpu_free_gb = self._gpu_total_free_gb()
            print(f"Detected GPU free memory ≈ {gpu_free_gb:.2f} GB")

            # Load on GPU (preferred)
            print("🔄 Loading model on GPU (4-bit)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                offload_folder=self.offload_folder,
                low_cpu_mem_usage=True,
            )
            print("✅ Model loaded on GPU successfully!")

        except Exception as e:
            print("❌ Failed to load model on GPU. Traceback:")
            traceback.print_exc()

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)



if __name__ == "__main__":
    llm = LLMInterface(model_name="./resources/gemma-2-9b-it", use_4bit=True)
    llm.load_model()
    print("Loaded the model on GPU")

    # Test generation
    prompt = "Explain quantum computing in simple terms."
    print("Generating the response")
    response = llm.generate(
        prompt,
        max_new_tokens=300,   # longer response
        temperature=0.7,      # creativity
        top_p=0.9,            # diversity
        do_sample=True
    )
    print("🗨️ Response:", response)
