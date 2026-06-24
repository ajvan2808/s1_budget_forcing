"""
Loader cho các model được dùng trong thực nghiệm.
Hỗ trợ:
  - 4-bit quantization (BitsAndBytes) để chạy trên GPU nhỏ (T4/A100 16GB)
  - TPU (torch_xla) — tự động detect, disable quantization trên TPU
  - CPU fallback
"""

from __future__ import annotations
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# TPU detection (optional; gracefully skip if torch_xla not installed)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

# ── Supported models ──────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    # Model đã được fine-tune với reasoning traces (dùng trực tiếp + BF)
    "s1-32B":           "simplescaling/s1-32B",
    "r1-distill-14B":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "r1-distill-7B":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "phi4-reasoning":   "microsoft/phi-4",  # Phi-4 has native reasoning capabilities

    # Base models (cần SFT trên s1K trước khi dùng BF)
    "qwen2.5-7B":       "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3B":       "Qwen/Qwen2.5-3B-Instruct",
    "llama3-8B":        "meta-llama/Meta-Llama-3-8B",
    "gemma4-E2B-it":    "google/gemma-4-E2B-it",

    # Vietnamese-specialized models (Sprint 3+)
    "vinallama-7b":     "vilm/vinallama-7b-chat",
    "vistral-7b":       "Viet-Mistral/Vistral-7B-Chat",   # canonical repo (vilm/vistral-7b-chat is mirror)
    "seallm-7b":        "SeaLLMs/SeaLLMs-v3-7B-Chat",
    # Vietnamese reasoning models (Sprint 4+)
    # GreenMind: GRPO-trained reasoning model, <think></think><answer></answer> format
    "greenmind-14b-r1": "GreenNode/GreenMind-Medium-14B-R1",
}


def detect_device() -> str:
    """
    Detect available device: TPU > CUDA > MPS > CPU.
    
    Returns:
        - "xla" for TPU (torch_xla available)
        - "cuda" for NVIDIA GPU
        - "mps" for Apple Metal Performance Shaders
        - "cpu" as fallback
    """
    if HAS_TPU:
        try:
            xm.get_ordinal()  # Check if TPU is actually accessible
            print("[device] TPU detected (torch_xla available)")
            return "xla"
        except Exception:
            pass
    
    if torch.cuda.is_available():
        print(f"[device] CUDA detected ({torch.cuda.get_device_name(0)})")
        return "cuda"
    
    if torch.backends.mps.is_available():
        print("[device] Metal Performance Shaders (MPS) detected")
        return "mps"
    
    print("[device] Falling back to CPU")
    return "cpu"


def get_bnb_config(load_in_4bit: bool = True, device_type: str = "cuda") -> BitsAndBytesConfig | None:
    """
    Trả về BitsAndBytesConfig cho 4-bit quantization.
    
    NOTE: TPU không hỗ trợ BitsAndBytes quantization → return None khi device_type="xla"
    """
    if not load_in_4bit or device_type == "xla":
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # QLoRA double quant
    )



def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> tuple:
    """
    Load model + tokenizer với optional 4-bit quantization.
    Tự động detect TPU, CUDA, MPS, CPU — disable quantization trên TPU.

    Args:
        model_name:   Key từ SUPPORTED_MODELS hoặc HuggingFace model ID
        load_in_4bit: True để dùng NF4 quantization (tiết kiệm VRAM) — ignored on TPU
        device_map:   "auto" tự phân bổ GPU/CPU; set to "sequential" on TPU
        cache_dir:    Thư mục cache local (hữu ích trên Colab)

    Returns:
        (model, tokenizer)

    Ví dụ:
        model, tokenizer = load_model_and_tokenizer("qwen2.5-7B")
        model, tokenizer = load_model_and_tokenizer("r1-distill-7B", load_in_4bit=True)
        # On TPU: automatically uses full precision (bfloat16)
    """
    # Detect device
    device_type = detect_device()
    
    # Resolve model ID
    hf_id = SUPPORTED_MODELS.get(model_name, model_name)
    
    # On TPU, disable quantization and adjust device_map
    if device_type == "xla":
        print(f"Loading: {hf_id} | 4-bit=False (TPU detected, quantization disabled)")
        load_in_4bit = False
        device_map = "sequential"
    else:
        print(f"Loading: {hf_id} | 4-bit={load_in_4bit} | device={device_type}")

    bnb_config = get_bnb_config(load_in_4bit, device_type)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        low_cpu_mem_usage=True,  # load shards directly to device — reduces peak memory spike
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Parameters: ~{model.num_parameters()/1e9:.1f}B")
    print(f"Device type: {device_type}")
    return model, tokenizer


def estimate_vram_gb(model_name: str, load_in_4bit: bool = True) -> float:
    """Ước tính VRAM cần thiết (GB) — dùng để check trước khi load."""
    # Approximate params count
    param_billions = {
        "s1-32B": 32, "r1-distill-14B": 14, "r1-distill-7B": 7,
        "qwen2.5-7B": 7, "qwen2.5-3B": 3, "llama3-8B": 8,
        "gemma4-E2B-it": 2, "phi4-reasoning": 14,
        # Vietnamese-specialized
        "vinallama-7b": 7, "vistral-7b": 7, "seallm-7b": 7,
        # Vietnamese reasoning
        "greenmind-14b-r1": 14,
    }
    b = param_billions.get(model_name, 7)
    bytes_per_param = 0.5 if load_in_4bit else 2  # 4-bit = 0.5 byte, bf16 = 2 bytes
    return b * 1e9 * bytes_per_param / 1e9  # in GB


if __name__ == "__main__":
    print("VRAM estimates (4-bit):")
    for name in SUPPORTED_MODELS:
        vram = estimate_vram_gb(name, load_in_4bit=True)
        print(f"  {name}: ~{vram:.0f} GB")
