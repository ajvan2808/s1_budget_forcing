# Report Outline: Budget Forcing Generalizability Study

Updated: 2026-06-11

## Working Title

**Does Budget Forcing Generalize Beyond Qwen? A Small-Scale Test-Time Scaling Study Across Open-Source Model Families**

## Central Claim to Test

The s1 paper shows that Budget Forcing can improve test-time scaling on Qwen-family reasoning models. This report tests whether that behavior transfers to other open-source model families under a constrained but reproducible experimental setup.

## 1. Introduction

- Test-time scaling motivation.
- Why Budget Forcing is attractive: simple decoding intervention, no retraining required for inference-only tests.
- Gap 1: limited evidence outside Qwen-family models.
- Research question and contributions.

## 2. Background: s1 and Budget Forcing

- s1K data curation overview.
- Budget Forcing mechanics: enforce minimum with `Wait`, enforce maximum with final-answer transition.
- Metrics: control, scaling, performance.
- Limits reported by the paper: model-family coverage, context window, repetitive loops.

## 3. Methodology

- Cross-family design: Qwen reference vs Gemma/Phi non-Qwen families.
- Benchmarks: GSM8K, ARC-Challenge, MATH500 when runtime allows.
- Budget sweep: `n_wait = 0, 1, 2` for small runs; optional `4`.
- Keep trigger fixed as `Wait` to isolate model-family effects.
- Artifact path: JSON outputs -> `summary_phase2.csv` / `summary_phase2.md` -> report tables.

## 4. Implementation

- `model_loader.py`: supported model registry.
- `run_eval.py`: benchmark registry, prompt formatting, answer extraction, metric computation.
- `run_phase2_generalizability.sh`: matrix orchestration.
- `summary_phase2.py`: aggregation.
- Runtime assumptions and hardware constraints.

## 5. Results

Populate only from real artifacts.

Required tables:

| Table | Source |
| --- | --- |
| Model/benchmark/n_wait accuracy table | `summary_phase2.csv` |
| Scaling and performance by model family | `summary_phase2.csv` |
| Runtime/blocker log | sprint progress docs |

Required figures when artifacts exist:

- Accuracy vs `n_wait` per model.
- Generalizability heatmap by model family and benchmark.

## 6. Discussion

- Does positive scaling appear outside Qwen?
- Are gains benchmark-specific?
- What failure modes appear: extraction errors, repetitive continuation, low instruction-following, model loading/hardware constraints?
- How should negative or mixed findings be interpreted?

## 7. Limitations

- Small sample sizes in smoke runs.
- Model download and hardware constraints.
- Some models may be instruction-tuned but not reasoning-specialized.
- Control metric is approximate unless target-vs-actual token budgets are logged explicitly.
- Trigger optimization is deferred.

## 8. Conclusion

- Answer Gap 1 using observed artifacts only.
- Identify whether the next phase should be broader runs, trigger optimization, or anti-repetition work.

## References

- s1: Simple Test-Time Scaling.
- DeepSeek-R1 technical report.
- Qwen/Gemma/Phi model documentation or papers.
- Benchmark dataset references.
