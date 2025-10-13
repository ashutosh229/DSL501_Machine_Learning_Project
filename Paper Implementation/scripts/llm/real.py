from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging
import torch
logging.set_verbosity_error() 

class LLMInterface:
    def __init__(self, model_name: str = "google/gemma-2-9b-it", use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16, 
            llm_int8_enable_fp32_cpu_offload=True
        )
        
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.1) -> str:
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
if __name__ == "__main__":
    llm = LLMInterface(model_name="./resources/gemma-2-9b-it", use_4bit=True)
    print("🔄 Loading model...")
    llm.load_model()
    print("✅ Model loaded successfully!")

    # Optional: test generation
    prompt = "Explain quantum computing in simple terms."
    print("🧠 Generating response...")
    output = llm.generate(prompt, max_new_tokens=50)
    print("🗨️ Response:", output)
