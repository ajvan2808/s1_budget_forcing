# s1: Simple Test-Time Scaling — Analysis, Review, and a Vietnamese Side Experiment

**Format:** 18 slides + 4 appendix backup slides · ~25–30 min talk
**Framing:** Paper analysis & review (main) + our own side experiment (supporting)
**Audience:** thesis committee, NLP/AI peers

---

## PART 1 — THE s1 PAPER (slides 1–9)

---

## Slide 1 — Title

**Title:** s1: Simple Test-Time Scaling
**Subtitle:** Paper Analysis, Review, and a Vietnamese Side Experiment

**Author:** [Your name] · UIT Master's Thesis · 2026

**Reference:** Muennighoff et al., 2025 — *s1: Simple Test-Time Scaling*

---

## Slide 2 — Motivation: Why Test-Time Scaling?

**Headline:** Scaling laws are hitting diminishing returns — can inference-time compute be the next frontier?

**Content:**

- Classic scaling: more parameters + more data = better models (Chinchilla, GPT-4)
- Problem: training costs are enormous and improvements are slowing
- New direction: **test-time scaling** — spend more compute *at inference*, not training
- OpenAI o1 demonstrated this publicly — but the recipe is closed
- Question: **what is the simplest open recipe to achieve test-time scaling?**
- s1 answers this with three components: data curation, fine-tuning, and Budget Forcing

**Visual:** graph showing training-time vs test-time scaling paradigm shift

---

## Slide 3 — The s1 Paper at a Glance

**Headline:** One model, 1K examples, one decoding trick — competitive with o1-preview on math

**Content:**

**What s1 achieves:**

- s1-32B (fine-tuned Qwen2.5-32B) outperforms o1-preview on MATH500 and AIME24
- Uses only 1,000 training examples (s1K)
- Budget Forcing at inference adds sequential test-time scaling

**The full recipe:**

```
59K reasoning problems
        ↓  quality + difficulty + diversity filtering
      s1K (1,000 problems)
        ↓  supervised fine-tuning
   s1-32B model
        ↓  Budget Forcing at inference
   test-time scaling
```

**Visual:** the three-stage pipeline diagram

**Speaker note:** The interesting claim is that *quality beats quantity* — 1K carefully chosen examples beat naive scaling of data

---

## Slide 4 — s1K: Data Curation

**Headline:** The key insight: careful selection of 1,000 examples matters more than having 59,000

**Content:**

**Three filtering stages:**

1. **Quality** — remove low-quality problems and reasoning traces
   - Filter by source reliability and solution correctness

2. **Difficulty** — keep problems that challenge the model
   - Use model performance as a signal: easy problems add little value
   - Use reasoning trace *length* as a difficulty proxy — harder problems generate longer traces

3. **Diversity** — ensure broad domain coverage
   - Domain labels from 50+ categories (math, science, code, logic...)
   - Prevent over-representation of any single topic

**Reasoning traces:** distilled from Gemini Thinking — long, detailed, explicit reasoning steps

**Key claim:** "The difficulty and diversity of the 1K examples matter more than the quantity"

**Visual:** funnel diagram: 59K → quality filter → difficulty filter → diversity filter → 1K

---

## Slide 5 — Budget Forcing: The Core Mechanism

**Headline:** BF forces the model to think longer — or shorter — at decoding time

**Content:**

**Two directions:**

**Enforce minimum (extend thinking):**

```
<think>
  Step 1: ...  [model tries to stop → EoT token]
  [BF: suppress EoT, append "Wait"]
  Wait... let me reconsider.
  Step 2: ...  [continues reasoning]
</think>
```

**Enforce maximum (cut thinking):**

```
<think>
  [after N tokens → inject "Final Answer:"]
</think>
Final Answer: 42
```

**Parameter:** `n_wait` = number of times EoT is suppressed

- `n_wait = 0` → greedy baseline (no BF)
- `n_wait = 1, 2` → forced extension

**Why it works:** suppressing premature stopping forces the model to *revisit* its reasoning — sometimes catching errors it would otherwise commit to

**Visual:** annotated generation trace with EoT and trigger highlighted

---

## Slide 6 — s1 Metrics: How to Measure Test-Time Scaling

**Headline:** s1 separates mechanical control from useful reasoning gain

**Content:**

s1 defines three evaluation axes — we adopt these for our own experiment:

| Metric | Definition | What it reveals |
|--|--|--|
| **Control** | Does thinking token count increase with budget? | Mechanism is working |
| **Scaling** | Does accuracy improve as budget increases? | Useful reasoning gain |
| **Performance** | Max accuracy across all budgets | Best achievable result |

**Key distinction:**

- A model may show good *Control* (tokens go up) but poor *Scaling* (accuracy stays flat)
- This separation is crucial — it distinguishes "the mechanism fires" from "the mechanism helps"

**Visual:** two line graphs side by side: one showing tokens↑ (control), one showing accuracy↑ (scaling)

**Speaker note:** We use exactly these three concepts in our Vietnamese experiment

---

## Slide 7 — s1 Results: What the Paper Claims

**Headline:** s1-32B achieves state-of-the-art open reasoning with minimal data

**Content:**

**Key results from the paper:**

- AIME24: 56.7% (vs o1-preview: 44.6%)
- MATH500: 96.2%
- GPQA Diamond: 59.6%

**Budget Forcing findings:**

- Positive test-time scaling on all three English reasoning benchmarks
- Accuracy increases monotonically with n_wait on AIME24
- Scaling eventually plateaus or degrades due to reasoning loops at very high budgets

**Sample efficiency:**

- s1-32B with 1K examples outperforms models trained on 800K+ examples (Sky-T1)
- Suggests data quality > data quantity for reasoning

**Visual:** comparison table from the paper — s1 vs o1-mini vs o1-preview vs Sky-T1

---

## Slide 8 — Critical Analysis: What s1 Does Not Test

**Headline:** s1's evaluation is English-centric and model-family-specific

**Content:**

**What s1 shows well:**

- BF works on English math/science reasoning benchmarks
- BF works on Qwen-family models fine-tuned with "Wait"-style traces
- Data curation quality matters more than quantity

**What s1 leaves open:**

1. **Language generalization** — all benchmarks are English. Does BF work in other languages?

2. **Model family generalization** — only Qwen/s1 models tested. Does BF require a specific training format?

3. **Task type generalization** — mainly math and science. Does BF help factual knowledge retrieval?

4. **Training paradigm** — s1 uses SFT. What happens with RLHF/GRPO-trained reasoning models?

**These open questions motivate our side experiment.**

**Visual:** four question marks mapped to each gap

---

## Slide 9 — s1 Summary

**Headline:** A clean, reproducible recipe — but tested under narrow conditions

**Content:**

**What s1 contributes:**

- Shows test-time scaling is achievable without massive resources
- Budget Forcing is simple, model-agnostic in principle, and requires no retraining
- s1K shows that 1,000 well-chosen examples can drive strong reasoning performance

**The open question we pick up:**
> Budget Forcing is presented as a general decoding-time method. But it was only tested on English benchmarks with Qwen models trained on "Wait"-style reasoning traces. **How general is it really?**

**Transition:** We designed a small side experiment to probe this — testing BF on Vietnamese models and benchmarks across different model families and training paradigms.

**Visual:** s1 recipe box → arrow → "Does this transfer?" → our experiment

---

## PART 2 — OUR VIETNAMESE SIDE EXPERIMENT (slides 10–17)

---

## Slide 10 — Our Side Experiment: Design

**Headline:** We test BF on Vietnamese — varying language, model family, and task type simultaneously

**Content:**

**Model matrix (2×2 across two axes):**

|  | English-primary | Vietnamese-primary |
|--|--|--|
| **Non-reasoning** | qwen2.5-3B | vinallama-7b · vistral-7b · seallm-7b |
| **Reasoning** | r1-distill-7B | greenmind-14b-r1 |

**Three Vietnamese benchmarks:**

| Benchmark | Type | Answer format |
|--|--|--|
| vi_gsm8k | Multi-step math | Numeric |
| vimmlu | Factual knowledge MC | A/B/C/D |
| vnhsge | HS exam MC | A/B/C/D |

**BF adaptation:**

- Trigger: *"Chờ một chút"* (Vietnamese for "Wait")
- n_wait ∈ {0, 1, 2} — n_wait=0 is greedy baseline
- n=100 samples per condition

**Visual:** the 2×2 matrix with arrows to the three benchmarks

---

## Slide 11 — Results Overview

**Headline:** BF outcome depends heavily on model training paradigm — not on language

**Content:** (introduce Fig 3 — BF delta bar chart)

| Model | vi_gsm8k | vimmlu | vnhsge | Pattern |
|--|--|--|--|--|
| r1-distill-7B | +4pp | +4pp | +3pp | ✓ Consistent gain |
| greenmind-14b-r1 | **−25pp** | −10pp | 0pp | ✗ Disrupted |
| seallm-7b | −19pp | +10pp | +6pp | ◑ Task-type split |
| vinallama-7b | −9pp | −13pp | −5pp | ✗ Degraded |
| vistral-7b | −5pp | −26pp | −33pp | ✗ Severely degraded |

**Visual:** Fig 3 (BF accuracy delta bar chart)

---

## Slide 12 — Observation 1: r1-distill Confirms BF Transfers

**Headline:** English reasoning model → positive BF in Vietnamese. Language is not the barrier.

**Content:**

- r1-distill-7B: distilled from DeepSeek-R1 with "Wait"-style reasoning traces
- Shows +3–4pp accuracy gain across all three Vietnamese benchmarks
- Thinking tokens increase steadily (191 → 426 on math)
- Modest but consistent — the same *direction* as s1's English results

**What this tells us about s1:**
> BF's mechanism — forcing the model to revisit reasoning before committing — is language-agnostic when the model was trained to use it. The model's internal reasoning structure works in Vietnamese even though it was trained on English.

**Visual:** Fig 1 — r1 accuracy line across three benchmarks

---

## Slide 13 — Observation 2: GreenMind — When BF Backfires

**Headline:** The most capable Vietnamese model — but BF drops math accuracy by 25pp

**Content:**

**GreenMind-14B-R1:**

- Native Vietnamese reasoning model, trained with GRPO (reward-based, not SFT)
- Produces `<think>...</think><answer>...</answer>` format natively
- **Baseline: 80% on vi_gsm8k** — strongest in our study

**Under BF:**

- Math: 80% → 55% (−25pp)
- Factual MC: 55% → 45% (−10pp)
- HS exam: 57% → 57% (0pp)

**The anomaly:** thinking tokens *decrease* at nwait=1 (199 → 145) before recovering — BF is shrinking thinking, not extending it

**Why:** GRPO training optimizes for compact, confident answers. BF's "Chờ một chút" injection breaks the `</think><answer>` structural handoff, corrupting the model's committed answer.

**What this tells us about s1:**
> BF requires a specific training format — "Wait"-style SFT traces. A different self-correction paradigm (GRPO) is incompatible, even when it produces better baseline accuracy.

**Visual:** Fig 2 — highlight greenmind's token count decrease at nwait=1

---

## Slide 14 — Observation 3: SeaLLM — Task Type Sets the Ceiling

**Headline:** Same model, opposite BF effects depending on what the task requires

**Content:**

**SeaLLM-7b on BF:**

- vi_gsm8k (math): 66% → 47% — **−19pp**
- vimmlu (factual MC): 46% → 56% — **+10pp**
- vnhsge (HS exam MC): 47% → 53% — **+6pp**

**Why the split?**

- **MC tasks:** BF gives the model a second pass → it can reconsider its letter choice → sometimes corrects wrong answers
- **Math tasks:** the arithmetic was already computed correctly → BF forces continuation → corrupts the extracted number

**What this tells us about s1:**
> s1 tested BF primarily on reasoning-heavy math benchmarks. Our result suggests BF's benefit depends on whether the task rewards *reconsideration*. On knowledge retrieval with multiple choice, the benefit appears even for models without explicit reasoning training.

**Visual:** side-by-side: SeaLLM accuracy lines on vi_gsm8k vs vimmlu

---

## Slide 15 — What We Observed: Three Lessons for s1

**Headline:** BF generalizes — but conditionally

**Content:**

| Condition | s1's assumption | What we found |
|--|--|--|
| Language | Tested English only | Language is **not** the barrier — r1 works in Vietnamese |
| Training format | SFT + "Wait" traces | **Critical** — GRPO-trained models (greenmind) are incompatible |
| Task type | Math/science reasoning | MC tasks also benefit — arithmetic tasks suffer under BF |

**The refined picture of BF:**

> Budget Forcing is **not** a universal decoding trick. It is a **training-format-dependent** mechanism. Models trained to use "Wait" as a self-correction signal absorb it. Models trained differently do not — regardless of how capable they are.

**Visual:** 2×2 grid: language × training format, colored by BF outcome

---

## Slide 16 — Limitations of Our Experiment

**Headline:** What our side experiment cannot claim

**Content:**

- **Sample size:** n=100 per condition — sufficient for directional signals, not for statistical confidence intervals
- **Single trigger phrase:** only *"Chờ một chút"* tested — other Vietnamese trigger variants may behave differently
- **No SFT baseline:** we test BF without fine-tuning Vietnamese models on s1K-style data — the combination of Vietnamese SFT + BF is untested
- **Automated evaluation:** answer extraction may slightly undercount accuracy for models that embed answers in full sentences
- **Flip analysis pending:** self-correction vs corruption rates at sample level not yet computed (requires full-trace re-run)

**Visual:** simple list with icons

---

## Slide 17 — Conclusion

**Headline:** s1 is a strong, reproducible test-time scaling recipe — with important boundary conditions

**Content:**

**On the s1 paper:**

- Budget Forcing is elegant and effective — no retraining required, just decoding-time control
- s1K shows that 1,000 well-curated reasoning examples can drive strong performance
- The recipe is genuinely open and reproducible

**What our experiment adds:**

- BF transfers across language (confirmed on Vietnamese with r1-distill)
- BF does **not** transfer across training paradigms — GRPO-trained models are incompatible
- Task type shapes BF's ceiling — reconsideration tasks (MC) benefit more than recomputation tasks (math)

**One-sentence takeaway:**
> The key condition for test-time scaling via Budget Forcing is not what language the model speaks — it is whether the model was trained to use "thinking continuation" as a self-correction signal.

**Visual:** Fig 3 as background — the full picture at a glance

---

## Slide 18 — Future Directions & Thank You

**Headline:** Where this leads

**Content:**

**Immediate next steps:**

- Fine-tune a Vietnamese model on s1K-style "Wait" traces → test BF compatibility
- Apply BF to GreenMind *after* SFT alignment — does it recover the 25pp drop?
- Test alternative Vietnamese triggers: *"Đợi đã"*, *"Suy nghĩ thêm"*

**Broader questions:**

- Can GRPO-trained models be made BF-compatible with a short adapter fine-tune?
- What is the minimum amount of "Wait"-style data needed for BF to work?
- Create Vi-s1K — a Vietnamese version of the s1K dataset

---

**Thank you. Questions?**

---

## Appendix Slides (backup for Q&A)

### A1 — Full Results Table

All 45 cells from summary_vi.csv (5 models × 3 benchmarks × 3 n_wait)

### A2 — s1K Curation Details

59K source breakdown, quality/difficulty/diversity filter specifics, Gemini Thinking trace distillation

### A3 — Benchmark Details

vi_gsm8k / vimmlu / vnhsge — format, sample counts, answer format conversion

### A4 — GreenMind Thinking Token Anomaly

Fig 2 close-up — why tokens decrease at nwait=1, GRPO training incompatibility explained
