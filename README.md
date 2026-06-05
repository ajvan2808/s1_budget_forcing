# s1 Budget Forcing — Đồ án Nhóm

Phân tích và thực nghiệm kỹ thuật **Budget Forcing** từ bài báo:
> *s1: Simple Test-Time Scaling* (EMNLP 2025)

---

## Cấu trúc Project

```md
s1_budget_forcing_project/
├── README.md                          ← File này
├── project_plan_s1_budget_forcing.md  ← Kế hoạch 2 tuần chi tiết
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
│
├── report/                            ── NGƯỜI B chủ trì
│   ├── outline.md                     ← Cấu trúc báo cáo
│   └── sections/                      ← Draft từng section
│
└── experiments/                       ── NGƯỜI A chủ trì
    ├── data/
    │   └── download_s1k.py            ← Download s1K dataset
    ├── models/
    │   └── model_loader.py            ← Load model 4-bit
    ├── budget_forcing/
    │   ├── __init__.py
    │   ├── decoding.py                ← Core: BudgetForcingDecoder
    │   └── metrics.py                 ← Control / Scaling / Performance
    ├── evaluation/
    │   └── run_eval.py                ← Main evaluation pipeline
    ├── notebooks/
    │   ├── 01_baseline_reproduction.ipynb
    │   └── 02_trigger_ablation.ipynb
    ├── scripts/
    │   ├── run_baseline.sh
    │   └── run_budget_forcing.sh
    └── results/                       ← JSON outputs + figures
```

## Quickstart (Người A)

```bash
# Using uv
uv sync

# Using python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Download data
python experiments/data/download_s1k.py

# 2. Chạy baseline
bash experiments/scripts/run_baseline.sh

# 3. Chạy Budget Forcing
bash experiments/scripts/run_budget_forcing.sh

# 4. Mở notebooks để phân tích
jupyter notebook experiments/notebooks/
```

## Key Concepts

| Khái niệm | Mô tả |
| ----------- | ------- |
| **Budget Forcing** | Can thiệp tại decoding: suppress EoT token + append "Wait" |
| **Enforce Minimum** | Append "Wait" khi model muốn dừng → tăng compute |
| **Enforce Maximum** | Chèn "Final Answer:" khi đạt token limit → giảm compute |
| **Control** | Tỷ lệ đạt đúng target compute (BF = 100%) |
| **Scaling** | Slope của accuracy-vs-compute (BF = +15, RS = -35) |
| **Performance** | Accuracy tối đa đạt được |

## Resources

- **Paper:** <https://arxiv.org/abs/2501.12599>
- **Code:** <https://github.com/simplescaling/s1>
- **Dataset:** `huggingface-cli download simplescaling/s1K`
