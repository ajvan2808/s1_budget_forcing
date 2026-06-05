# Kế hoạch Đồ án Nhóm: Phân tích & Thực nghiệm s1 Test-Time Scaling
>
> **Nhóm:** 2 người | **Timeline:** 2 tuần (14 ngày) | **Hoàn thành:** Tuần 2, Ngày 14

---

## Tổng quan Dự án

| Hạng mục | Chi tiết |
| ---------- | ---------- |
| **Bài báo chính** | *s1: Simple Test-Time Scaling* (EMNLP 2025) |
| **Bài báo tham khảo** | *RAG for Educational Application: A Systematic Survey* (C&E:AI 2025) |
| **Deliverables** | Báo cáo phân tích + Demo thực nghiệm Budget Forcing |
| **Repo** | `s1_budget_forcing_project/` |

---

## Phân công Nhân sự

| | **Người A** | **Người B** |
| --- | ------------- | ------------- |
| **Vai trò chính** | Thực nghiệm (Experiments) | Báo cáo (Report) |
| **Thế mạnh** | Python/ML, GPU setup | Viết, phân tích, tổng hợp |
| **Deliverable** | Code + kết quả thực nghiệm | Draft báo cáo đầy đủ |

> **Nguyên tắc phân công:** Tuần 1 làm song song gần như hoàn toàn độc lập. Tuần 2 tích hợp kết quả vào báo cáo, cùng review.

---

## Timeline Chi tiết

### 🗓 TUẦN 1 (Ngày 1–7): Setup + Core Work

#### Ngày 1–2 | Setup Song song (Độc lập 100%)

| Người A — Thực nghiệm | Người B — Báo cáo |
| ---------------------- | ------------------- |
| Clone repo s1 từ GitHub | Đọc toàn bộ paper s1 + file phân tích .md |
| Setup Python environment (`requirements.txt`) | Setup cấu trúc báo cáo (outline, sections) |
| Download dataset s1K (`huggingface datasets`) | Expand phần Giới thiệu & Tóm tắt bài báo |
| Test load model Qwen2.5-7B (quantized 4-bit) | Tìm thêm 3–5 bài báo liên quan (related work) |
| **Checkpoint:** Environment chạy được, model load OK | **Checkpoint:** Outline báo cáo draft + Section 1 draft |

#### Ngày 3–4 | Implement Core (Song song)

| Người A | Người B |
| --------- | --------- |
| Implement `BudgetForcingDecoder` (suppress EoT + append "Wait") | Viết Section 2: Phân tích phương pháp s1K data curation |
| Implement `EnforceMaximum` (insert Final Answer delimiter) | Viết Section 3: Phân tích Budget Forcing (cơ chế, metrics) |
| Unit test: verify token suppression hoạt động đúng | Viết Section 4: Điểm mạnh/yếu, research gaps |
| **Checkpoint:** Budget Forcing chạy được trên 5 sample | **Checkpoint:** Section 2–4 draft (rough) |

#### Ngày 5 | Baseline Evaluation

| Người A | Người B |
| --------- | --------- |
| Chạy greedy baseline (no BF) trên 50 câu MATH500 | Viết Section 5: Related Work (so sánh với RAG survey) |
| Chạy BF với Wait×1, Wait×2 trên cùng 50 câu | Tạo bảng so sánh các phương pháp (Table template) |
| Ghi kết quả vào `results/baseline.json` | Viết phần Methodology trong báo cáo |
| **Checkpoint:** Có số liệu baseline đầu tiên | **Checkpoint:** Section 5 + Methodology draft |

#### Ngày 6–7 | Mở rộng Experiments + Review Nội bộ

| Người A | Người B |
| --------- | --------- |
| Chạy full 200 câu MATH500 (baseline + BF ×1/2/4) | Review toàn bộ draft tuần 1, thống nhất format |
| Tính 3 metrics: Control, Scaling, Performance | Tạo Figure templates (matplotlib) cho plots |
| Plot accuracy vs. compute curve | Viết tóm tắt kết quả *dự kiến* để A điền số |
| **Checkpoint:** Draft kết quả tuần 1 | **Checkpoint:** Báo cáo 60% hoàn thành |

---

### 🗓 TUẦN 2 (Ngày 8–14): Ablation + Tích hợp + Hoàn thiện

#### Ngày 8–9 | Ablation Studies + Viết Experiments

| Người A | Người B |
| --------- | --------- |
| Ablation 1: Test trigger phrases ("Hmm", "Let me reconsider", "Actually") | Điền kết quả tuần 1 vào Section Experiments |
| So sánh với Class-Conditional Control (CCC) baseline | Viết analysis, interpret kết quả A gửi |
| Ghi kết quả ablation vào `results/ablation_triggers.json` | Viết Section 6: Discussion |
| **Checkpoint:** Ablation trigger data | **Checkpoint:** Section Experiments draft |

#### Ngày 10–11 | Error Analysis + Cross-check

| Người A | Người B |
| --------- | --------- |
| Phân tích lỗi: case study 5–10 câu BF đúng vs. sai | Viết Section 7: Kết luận & Hướng mở rộng |
| Kiểm tra repetitive loop cases: đếm tỷ lệ | Cross-check toàn bộ số liệu A gửi với draft báo cáo |
| Tạo visualization: heatmap trigger × model accuracy | Cập nhật abstract, introduction với kết quả cuối |
| **Checkpoint:** Phân tích lỗi + visualizations | **Checkpoint:** Draft báo cáo 90% |

#### Ngày 12 | Integration Day (Cùng làm)

- A gửi toàn bộ figures, tables, số liệu cuối cho B
- B tích hợp vào báo cáo, format LaTeX/Word
- Review chéo: A đọc báo cáo, B review code/notebook
- Thống nhất narrative: story từ kết quả

#### Ngày 13 | Slides + Polish

| Người A | Người B |
| --------- | --------- |
| Tạo demo notebook: clean, có comments | Tạo slides presentation (10–12 slides) |
| Viết README cho project folder | Final proofread báo cáo |
| **Checkpoint:** Demo notebook sẵn sàng | **Checkpoint:** Slides draft |

#### Ngày 14 | Final Review + Submit

- Cả hai cùng review lần cuối
- Kiểm tra: references, figures, code chạy được
- Submit báo cáo + code repo

---

## Deliverables Checklist

### Báo cáo (Người B chủ trì, A review)

- [ ] Section 1: Giới thiệu & Bài toán nghiên cứu
- [ ] Section 2: Tóm tắt phương pháp s1 (data curation, SFT)
- [ ] Section 3: Phân tích Budget Forcing (cơ chế + 3 metrics)
- [ ] Section 4: Điểm mạnh / yếu / research gaps
- [ ] Section 5: Related work (so sánh với các hướng khác, liên hệ RAG survey)
- [ ] Section 6: Thực nghiệm (setup, kết quả, ablation)
- [ ] Section 7: Thảo luận & Kết luận
- [ ] References

### Code & Experiments (Người A chủ trì, B review)

- [ ] `budget_forcing/decoding.py` — BF implementation
- [ ] `evaluation/run_eval.py` — evaluation pipeline
- [ ] `results/` — JSON kết quả các experiments
- [ ] `notebooks/01_baseline_reproduction.ipynb` — reproduce baseline
- [ ] `notebooks/02_trigger_ablation.ipynb` — BF + trigger ablation + plots

### Scope tuần 2 (MVP để đảm bảo khả thi)

- [ ] Ưu tiên 1 benchmark chính (`math500`) với `n_wait = 0,1,2,4`
- [ ] Trigger ablation giới hạn 2 trigger (`Wait`, `Hmm, let me reconsider`)
- [ ] Chạy `n_samples=50` trước, mở rộng 200 mẫu nếu còn GPU/time
- [ ] Hoàn thiện báo cáo với kết quả reproducible trước khi mở rộng ý tưởng phụ

### Slides

- [ ] 12 slides: Motivation → Method → Experiments → Conclusion

---

## Dependency Map

```md
Ngày 1–5: A và B làm hoàn toàn SONG SONG
    A: env → implement BF → baseline eval
    B: outline → sections 2-4 → methodology

Ngày 6–7: A gửi DRAFT RESULTS cho B
    → B dùng để viết phần Experiments skeleton

Ngày 8–11: Song song tiếp
    A: ablation + error analysis
    B: hoàn thiện sections + tích hợp kết quả

Ngày 12: INTEGRATION POINT (blocking)
    → Cần A hoàn thành kết quả cuối trước khi B finalize

Ngày 13–14: Song song
    A: clean code + demo
    B: slides + final proofread
```

---

## Risk Management

| Rủi ro | Xác suất | Biện pháp |
| -------- | ---------- | ----------- |
| GPU/Colab quota hết | Cao | Dùng Qwen2.5-1.5B hoặc 3B để prototype nhanh; batch nhỏ 50 câu thay vì 500 |
| Model không load được (OOM) | Trung bình | 4-bit quantization bắt buộc; dùng `device_map="auto"` |
| BF không reproduce được | Thấp | Dùng s1-32B GGUF quantized từ HuggingFace thay vì self-train |
| Thời gian viết báo cáo vượt kế hoạch | Trung bình | B dùng file phân tích .md có sẵn làm xương sống; không cần viết lại từ đầu |
| Kết quả thực nghiệm kém (không reproduce) | Trung bình | Negative result vẫn có giá trị; phân tích nguyên nhân |

---

## Tài nguyên Tham khảo

| Tài nguyên | Link |
| ------------ | ------ |
| s1 GitHub | <https://github.com/simplescaling/s1> |
| s1K dataset | `huggingface-cli download simplescaling/s1K` |
| Qwen2.5-7B (4-bit) | `bartowski/Qwen2.5-7B-Instruct-GGUF` |
| lm-evaluation-harness | `https://github.com/EleutherAI/lm-evaluation-harness` |
| MATH500 | Tích hợp trong lm-eval-harness |
