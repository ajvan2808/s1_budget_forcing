"""
Download và chuẩn bị dataset s1K từ HuggingFace.
s1K = 1,000 mẫu được lọc theo Quality + Difficulty + Diversity.

Chạy: python download_s1k.py
"""

from datasets import load_dataset
import json
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def download_s1k(save_path: str = "s1k.json"):
    """Download s1K dataset từ HuggingFace."""
    print("Downloading s1K dataset...")
    ds = load_dataset("simplescaling/s1K", cache_dir=str(CACHE_DIR))
    print(f"Dataset loaded: {ds}")
    print(f"  Train split size: {len(ds['train'])}")
    print(f"  Columns: {ds['train'].column_names}")

    # Save local copy
    data = [dict(row) for row in ds["train"]]
    out = DATA_DIR / save_path
    with open(out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {out}")
    return data


def sample_subset(data: list, n: int = 100, seed: int = 42) -> list:
    """Lấy subset nhỏ để test nhanh."""
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def inspect_sample(data: list, idx: int = 0):
    """In ra 1 mẫu để kiểm tra format."""
    sample = data[idx]
    print(f"\n--- Sample {idx} ---")
    for k, v in sample.items():
        val_str = str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
        print(f"  {k}: {val_str}")


if __name__ == "__main__":
    data = download_s1k()
    inspect_sample(data, 0)
    inspect_sample(data, 1)

    # Tạo mini subset (50 samples) để test pipeline nhanh
    mini = sample_subset(data, n=50)
    mini_path = DATA_DIR / "s1k_mini50.json"
    with open(mini_path, "w") as f:
        json.dump(mini, f, indent=2, ensure_ascii=False)
    print(f"\nMini subset (50 samples) saved to {mini_path}")

    # Thống kê domain distribution
    if "domain" in data[0]:
        from collections import Counter
        domains = Counter(s.get("domain", "unknown") for s in data)
        print("\nTop 10 domains:")
        for d, c in domains.most_common(10):
            print(f"  {d}: {c}")
