# Agent B Handoff — Sprint 3

Role: Writer / Report owner
Updated: 2026-06-11

---

## Mission

Reframe the academic report for the Vietnamese BF+RAG topic.
You do NOT need experiment results to start — write from prior work + hypotheses.
Result tables use explicit placeholders until Sprint 4 artifacts exist.

---

## Starting State

Files that need full rewrite:
- `report/outline.md` — was: "Does BF Generalize Beyond Qwen?" → rewrite for Vi BF+RAG

Files that need targeted updates:
- `report/sections/01_introduction.tex` — new motivation paragraph
- `report/sections/02_method.tex` — add §2.3 RAG Augmentation
- `report/sections/05_related_work.tex` — add Vi NLP + RAG subsection
- `report/sections/06_experiments.tex` — add result table placeholders

Files that are already compatible (check, minor edits only):
- `report/sections/03_metrics.tex` — BF metrics unchanged; add note on 3-condition comparison
- `report/sections/04_analysis.tex` — placeholder analysis tables, mark as TBD

---

## Task Checklist

### W1 — Rewrite report/outline.md

New title: "Budget Forcing on Vietnamese Reasoning: Test-Time Scaling vs. Retrieval Augmentation"

8-section structure:
1. Introduction — Vi NLP context, BF gap, RAG comparison, contributions
2. Background — BF mechanics (s1 paper), RAG overview, Vietnamese reasoning benchmarks
3. Metrics — Control / Scaling / Performance (unchanged from §3 in prior outline)
4. Method — §4.1 BF Decoding, §4.2 RAG Pipeline, §4.3 BF+RAG Combined
5. Related Work — s1/BF papers, Vi NLP, RAG-for-reasoning, educational AI in Vietnam
6. Experiments — Setup, Results tables, Scaling curves
7. Discussion — BF vs RAG task analysis, combined effect, failure modes
8. Conclusion — Summary, limitations, future work (Vi SFT, harder benchmarks)

### W2 — Update 01_introduction.tex

New opening:
- Paragraph 1: Vietnamese as underrepresented language in reasoning benchmarks
- Paragraph 2: Two complementary scaling strategies — BF (thinking longer) vs RAG (knowing more)
- Paragraph 3: Research gap — no published BF vs RAG comparison on Vietnamese
- Paragraph 4: Our contributions (3 bullets)
  1. First BF evaluation on Vietnamese reasoning (MGSM-vi, ViMMLU)
  2. First BF vs RAG head-to-head on Vietnamese tasks
  3. BF+RAG combined evaluation

### W3 — Update 02_method.tex

Add §2.3 after existing BF sections:

```
\subsection{RAG Augmentation for Vietnamese Reasoning}
We augment each question with top-$k$ passages retrieved from Vietnamese Wikipedia
using a multilingual dense retriever...

Three evaluation conditions:
\begin{itemize}
  \item \textbf{BF only}: budget forcing with $n_{\text{wait}} \in \{0, 1, 2\}$, no retrieval
  \item \textbf{RAG only}: top-3 passages prepended, greedy decoding ($n_{\text{wait}}=0$)
  \item \textbf{BF+RAG}: top-3 passages prepended + budget forcing ($n_{\text{wait}} \in \{1, 2\}$)
\end{itemize}
```

### W4 — Update 05_related_work.tex

Add subsection: "Vietnamese Language Models and Benchmarks"
- PhoGPT, Vinallama, Vistral, SeaLLM — cite original HF model cards / papers
- ViMMLU benchmark paper
- MGSM multilingual evaluation (Shi et al., 2023)

Add subsection: "RAG for Reasoning and Education"
- RAG foundational (Lewis et al., 2020)
- RAG + reasoning (Shi et al., 2023 — distractor retrieval; He et al., 2022)
- Vietnamese educational AI context (1–2 sentences)

### W5 — Add placeholders in 06_experiments.tex

Main result table skeleton:
```
\begin{table}[h]
\centering
\caption{Accuracy by condition on Vietnamese benchmarks (\textsc{TBD})}
\begin{tabular}{lllccc}
\toprule
Model & Benchmark & Condition & n\_wait=0 & n\_wait=1 & n\_wait=2 \\
\midrule
qwen2.5-3B & vi\_gsm8k & BF\_only & -- & -- & -- \\
qwen2.5-3B & vi\_gsm8k & RAG\_only & -- & -- & -- \\
qwen2.5-3B & vi\_gsm8k & BF+RAG  & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

All `--` cells are replaced in Sprint 4 from `summary_vi.csv`.
Do not invent numbers.

---

## Key Writing Decisions

**Framing:** This is a comparison paper, not purely a BF paper.
Lead with the question "thinking longer vs knowing more" — more compelling than
"does BF work for Vietnamese."

**Hypothesis to articulate (before results):**
- BF > RAG on multi-step math (vi_gsm8k, ZaloAI): retrieved text rarely contains
  the calculation steps; extended thinking helps more.
- RAG ≥ BF on factual knowledge (ViMMLU): external knowledge directly resolves
  factual gaps that extra thinking cannot.
- BF+RAG ≥ max(BF, RAG) when context window is not saturated.

**Language:** Report in English. Vietnamese benchmark/model names stay as-is.

**Citations:** Do not cite papers you have not verified exist. Use `\cite{TODO:X}` as placeholder.

---

## Do Not Do (Writer Constraints)

- Do not fill result tables with plausible-looking numbers (keep all `--` or `TBD`)
- Do not claim specific accuracy values in the introduction prose
- Do not add new research questions beyond RQ1–RQ4 in PROJECT_BRIEF.md §2

---

## Handoff to Sprint 4

When Sprint 3 code is done and smoke run passes, Agent A will provide:
- `experiments/results/vi_{timestamp}/summary_vi.csv`

At that point, fill `--` cells in result tables and write the Discussion section.
