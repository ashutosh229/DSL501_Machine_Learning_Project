from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
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
            # returns (free, total) bytes for device 0
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024 ** 3)
        except Exception:
            return 0

    def load_model(self):
        # Prepare quantization config (only used if we decide to actually do 4-bit)
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # choose strategy based on GPU presence and free VRAM
        gpu_free_gb = self._gpu_total_free_gb()
        print(f"Detected GPU free memory ≈ {gpu_free_gb:.2f} GB")

        # Heuristic thresholds (adjust if you know your GPU)
        need_vram_for_9b_4bit = 8.0   # approximate minimum VRAM to hold 9B model in 4-bit
        try_gpu_4bit = self.use_4bit and (gpu_free_gb >= need_vram_for_9b_4bit)

        # Try path A: GPU + 4-bit (preferred if GPU big enough)
        if try_gpu_4bit:
            try:
                print("Attempting to load 4-bit model on GPU (device_map='auto')...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    offload_folder=self.offload_folder,   # allows safe temporary offload to disk
                    low_cpu_mem_usage=True,
                )
                print("Loaded model in 4-bit on GPU.")
                return
            except Exception as e:
                print("Failed to load 4-bit on GPU (falling back).")
                traceback.print_exc()

        # Path B: No suitable GPU or GPU 4-bit failed — load on CPU WITHOUT bitsandbytes quantization
        # NOTE: using 4-bit on CPU often triggers huge temporary allocations; avoid it.
        print("Loading model on CPU (disabling bitsandbytes quantization to avoid CPU OOM).")
        try:
            # Force no BitsAndBytes: set quantization_config=None and use low_cpu_mem_usage
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=None,     # IMPORTANT: disable bnb quantization here
                device_map={"": "cpu"},
                torch_dtype=torch.float32,    # use float32 on CPU for stability (or float16 if supported)
                low_cpu_mem_usage=True,
                offload_folder=self.offload_folder
            )
            print("Loaded model on CPU (float32) successfully.")
            return
        except Exception as e:
            print("CPU load failed — final fallback: try device_map='auto' with low_cpu_mem_usage and offload_folder.")
            traceback.print_exc()
            # final fallback attempt (may still fail if memory is too small)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config if self.use_4bit else None,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder=self.offload_folder
            )
    
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
