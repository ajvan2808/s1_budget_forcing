# Brainstorm

Updated: 2026-06-11

## Participants

- Coordinator: keeps scope narrow and artifacts traceable.
- Agent A (Experiment owner): prioritizes executable evidence, small runs before large sweeps.
- Agent B (Report owner): keeps the academic narrative coherent and aligned with available evidence.

---

## Scope History

### Phase 1 — Original: Reproduction only
Reproduce s1 BF baseline on MATH500. Superseded when Gap 1 was identified.

### Phase 2 — Gap 1: Cross-family generalizability (archived)
Budget Forcing on Qwen vs Gemma vs Phi vs Llama. Implementation complete; execution blocked on hardware.
Archived because a stronger, more novel direction was identified.

### Phase 3 — Current: Vietnamese BF Transfer Study (active)
**Decision rationale:**
- Gap 1 (cross-family BF) was already partially addressed by Yong et al. 2025 (cited in s1 Limitations). Lower novelty.
- BF transfer to Vietnamese has **zero published study**. High novelty.
- Real-world impact: Vietnamese math/science education tools.
- Inference-only → feasible on Kaggle T4.

**Scope confirmed (post-pivot, Sprint 3+):**
> "Does Test-Time Scaling via Budget Forcing Transfer to Vietnamese Language Reasoning?"

RAG was explored but is off-topic for this study. BF-only, n_wait sweep.

---

## Final Scope Decision (Sprint 3+)

**Budget Forcing on Vietnamese Reasoning — BF-only**

Single condition per (model, benchmark, n_wait):
- `BF_only` — decoding-time compute extension, no retrieval
- n_wait ∈ {0, 1, 2} where n_wait=0 is the greedy baseline

**Core question:** Does thinking longer via BF produce positive test-time scaling on Vietnamese benchmarks? Does this transfer across multilingual and Vietnamese-specialized model families?

---

## Benchmark Selection Rationale

| Benchmark | Type | Why |
|-----------|------|-----|
| `vi_gsm8k` (MGSM-vi) | Math reasoning | Confirmed on HF, parallel to English GSM8K |
| `vimmlu` | Multi-domain knowledge | Broad coverage; tests factual recall where RAG should shine |
| `zaloai_math` | Hard Vietnamese math | Domain-specific; tests reasoning depth where BF should shine |

**Hypothesis:** BF > RAG on vi_gsm8k (multi-step math; retrieved text less helpful). RAG ≥ BF on vimmlu (factual recall; external knowledge helps). BF+RAG ≥ max(BF, RAG) on both if prompts stay within context window.

---

## Model Selection Rationale

| Model | Type | Why |
|-------|------|-----|
| `qwen2.5-3B` | Multilingual instruction | Fast smoke-test baseline |
| `r1-distill-7B` | Reasoning-specialized (Qwen base + distillation) | Closest to paper's s1-32B setup |
| `vinallama-7b` | Vietnamese-specialized (LLaMA-2 base) | Does BF work on Vi-fine-tuned models? |
| `vistral-7b` | Vietnamese-specialized (Mistral base) | Second Vi comparison |
| `seallm-7b` | SEA multilingual | Regional multilingual baseline |

**Key design decision:** r1-distill-7B replaces qwen2.5-7B. Reasoning-specialized model is closer to the paper's s1-32B and creates a cleaner contrast: reasoning-specialized multilingual vs. instruction-tuned Vietnamese-specialized.

**Caveat:** Vinallama/Vistral (LLaMA-2/Mistral base) were not trained with long-CoT. Their EoT token is `</s>`, not `<|im_end|>`. BF response may differ — this is itself a finding.

Run order: qwen2.5-3B (smoke) → r1-distill-7B → vinallama-7b, vistral-7b, seallm-7b.

---

## Trigger Decision

Active trigger: **`"Chờ một chút"`** — Vietnamese, language-consistent with task.
Paper original used `"Wait"` (English). Using VN trigger removes language-mixing as a confound.

## Deferred Ideas (do not expand until main matrix is done)

- Trigger ablation: compare `"Chờ một chút"` vs `"Đợi đã"` vs `"Hãy suy nghĩ lại"`
- Anti-repetition controls
- Adaptive budget allocation by question difficulty
- SFT of Vi models on s1K-translated
- Human evaluation of reasoning trace quality in Vietnamese

---

## Practical Decision

Canonical smoke command:
```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf.sh
```

Use `summary_vi.py` as the canonical bridge from JSON → report tables.
