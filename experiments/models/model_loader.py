"""
Loader cho các model được dùng trong thực nghiệm.
Hỗ trợ 4-bit quantization (BitsAndBytes) để chạy trên GPU nhỏ (T4/A100 16GB).
"""

from __future__ import annotations
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── Supported models ──────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    # Model đã được fine-tune với reasoning traces (dùng trực tiếp + BF)
    "s1-32B":           "simplescaling/s1-32B",
    "r1-distill-14B":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "r1-distill-7B":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",

    # Base models (cần SFT trên s1K trước khi dùng BF)
    "qwen2.5-7B":       "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3B":       "Qwen/Qwen2.5-3B-Instruct",   # fallback khi VRAM < 8GB
    "llama3.1-8B":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def get_bnb_config(load_in_4bit: bool = True) -> BitsAndBytesConfig | None:
    """Trả về BitsAndBytesConfig cho 4-bit quantization."""
    if not load_in_4bit:
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

    Args:
        model_name:   Key từ SUPPORTED_MODELS hoặc HuggingFace model ID
        load_in_4bit: True để dùng NF4 quantization (tiết kiệm VRAM)
        device_map:   "auto" tự phân bổ GPU/CPU
        cache_dir:    Thư mục cache local (hữu ích trên Colab)

    Returns:
        (model, tokenizer)

    Ví dụ:
        model, tokenizer = load_model_and_tokenizer("qwen2.5-7B")
        model, tokenizer = load_model_and_tokenizer("r1-distill-14B", load_in_4bit=True)
    """
    # Resolve model ID
    hf_id = SUPPORTED_MODELS.get(model_name, model_name)
    print(f"Loading: {hf_id} | 4-bit={load_in_4bit}")

    bnb_config = get_bnb_config(load_in_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Parameters: ~{model.num_parameters()/1e9:.1f}B")
    return model, tokenizer


def estimate_vram_gb(model_name: str, load_in_4bit: bool = True) -> float:
    """Ước tính VRAM cần thiết (GB) — dùng để check trước khi load."""
    # Approximate params count
    param_billions = {
        "s1-32B": 32, "r1-distill-14B": 14, "r1-distill-7B": 7,
        "qwen2.5-7B": 7, "qwen2.5-3B": 3, "llama3.1-8B": 8,
    }
    b = param_billions.get(model_name, 7)
    bytes_per_param = 0.5 if load_in_4bit else 2  # 4-bit = 0.5 byte, bf16 = 2 bytes
    return b * 1e9 * bytes_per_param / 1e9  # in GB


if __name__ == "__main__":
    print("VRAM estimates (4-bit):")
    for name in SUPPORTED_MODELS:
        vram = estimate_vram_gb(name, load_in_4bit=True)
        print(f"  {name}: ~{vram:.0f} GB")
