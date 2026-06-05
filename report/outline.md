# Outline Báo cáo: Phân tích & Thực nghiệm s1 Test-Time Scaling

> **Người thực hiện:** Người B (chủ trì), Người A (review + cung cấp số liệu)
> **Timeline:** Draft tuần 1; Final tuần 2

---

## Cấu trúc Báo cáo (~15–20 trang)

### 1. Giới thiệu (1–2 trang)

- Motivation: Test-time scaling là gì? Tại sao quan trọng?
- OpenAI o1 và vấn đề thiếu minh bạch
- Đóng góp của bài báo s1: đơn giản + mã nguồn mở
- Cấu trúc báo cáo

**Nguồn:** Section 1 của s1 paper + file .md phân tích

---

### 2. Tóm tắt Phương pháp s1 (3–4 trang)

#### 2.1 Bài toán nghiên cứu

- Định nghĩa test-time scaling behavior
- Tại sao SFT + 1K samples đủ?

#### 2.2 Tạo dataset s1K

- 3 nguyên tắc: Quality → Difficulty → Diversity
- Quy trình lọc (59K → 1K)
- Đặc điểm: 53.6% sai nhưng vẫn tốt

#### 2.3 Budget Forcing

- Cơ chế Enforce Maximum và Minimum
- Token "Wait" và self-reflection
- So sánh với TCC, SCC, CCC, RS (Table 3)

**Nguồn:** Sections 2–3 của paper; Figure 3 (raspberry example)

---

### 3. Phân tích 3 Metrics Đánh giá (2 trang)

#### 3.1 Control (Eq. 1)

- Định nghĩa và lý do BF đạt 100%
- Tại sao TCC thất bại (model cannot count tokens)

#### 3.2 Scaling (Eq. 2)

- Positive slope → BF works; Negative slope → RS fails
- Giải thích tại sao RS có inverse scaling

#### 3.3 Performance (Eq. 3)

- Max accuracy; BF vượt o1-preview

**Nguồn:** Section 3.2, Table 3, Figure 6

---

### 4. Điểm mạnh, Hạn chế & Research Gaps (2 trang)

#### 4.1 Điểm mạnh

- Sample efficiency (1K vs 800K)
- Simplicity; Reproducibility; Open-source

#### 4.2 Điểm yếu

- Chỉ Qwen family; Context window; Repetitive loops; Math only

#### 4.3 Research Gaps (từ phân tích .md)

- Gap 2: Trigger optimization ← **đây là hướng thực nghiệm nhóm**
- Gap 5: Anti-repetition
- Gap 3: Non-math tasks

---

### 5. Related Work (1–2 trang)

#### 5.1 Test-Time Scaling Methods

- OpenAI o1; DeepSeek R1; Sky-T1
- MCTS, majority voting, best-of-N

#### 5.2 RAG trong Giáo dục (liên hệ bài RAG survey)

- RAG cũng là một hướng augment LLM tại inference time
- So sánh: RAG bổ sung knowledge bên ngoài; BF khai thác knowledge nội tại
- Điểm chung: đều không cần retrain model

---

### 6. Thực nghiệm (3–4 trang)

#### 6.1 Setup

- Model sử dụng, benchmark, hyperparameters
- Environment (GPU, quantization)

#### 6.2 Kết quả Baseline

- Accuracy without BF
- So sánh với paper (gap analysis)

#### 6.3 Budget Forcing Results

- Accuracy vs. n_wait (0, 1, 2, 4)
- Control / Scaling / Performance metrics
- Plot: accuracy-vs-compute curve

#### 6.4 Trigger Phrase Ablation

- Bảng so sánh 5 trigger phrases
- Best trigger cho model này
- Phân tích: productive reflection vs. repetitive loop

#### 6.5 Phân tích Lỗi (Error Analysis)

- Case studies: BF đúng ở đâu, sai ở đâu
- Repetitive loop cases

---

### 7. Thảo luận & Kết luận (1 trang)

#### 7.1 Findings chính

- BF có hoạt động không? Với mức độ nào?
- Trigger nào tốt nhất?
- Giới hạn thực nghiệm

#### 7.2 Kết luận

- Đóng góp của nhóm
- Hướng mở rộng tương lai

---

### References

- s1 paper (EMNLP 2025)
- RAG survey (C&E:AI 2025)
- DeepSeek R1; Qwen2.5; lm-evaluation-harness
- Budget Forcing related papers

---

## Ghi chú format

- LaTeX (recommended) hoặc Word
- Figures: matplotlib, 150+ DPI
- Tables: align với format paper gốc
- Code snippets: nếu cần minh họa, đưa vào appendix
