"""
3 Metrics đánh giá Test-Time Scaling theo paper s1 (Section 3.2):
  1. Control   — Mức độ kiểm soát lượng compute thực tế vs. target
  2. Scaling   — Slope của accuracy-vs-compute curve (dương = tốt)
  3. Performance — Accuracy tối đa đạt được

Tham khảo: Equations 1–3, Section 3.2.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ScalingMetrics:
    control: float | None  # 0.0 – 1.0, None = chưa đủ dữ liệu để tính
    scaling: float      # slope (dương = tốt; BF paper đạt +15)
    performance: float  # max accuracy (0.0 – 1.0)

    def __repr__(self):
        control_str = "N/A" if self.control is None else f"{self.control:.1%}"
        return (
            f"ScalingMetrics("
            f"control={control_str}, "
            f"scaling={self.scaling:+.1f}, "
            f"performance={self.performance:.1%})"
        )


def compute_control(
    target_budgets: Sequence[int],
    actual_tokens: Sequence[int],
) -> float:
    """
    Eq. 1 — Control: tỷ lệ runs đạt đúng budget target.

    Control = |{i : actual[i] ≈ target[i]}| / N

    'Đạt đúng' được định nghĩa là actual ∈ [0.8 × target, 1.2 × target].
    Budget Forcing đạt 100% vì nó can thiệp trực tiếp vào decoding.

    Args:
        target_budgets: Danh sách token budget mong muốn cho mỗi run
        actual_tokens:  Số token thinking thực tế sinh ra
    Returns:
        Control score ∈ [0, 1]
    """
    assert len(target_budgets) == len(actual_tokens), "Length mismatch"
    hits = sum(
        0.8 * t <= a <= 1.2 * t
        for t, a in zip(target_budgets, actual_tokens)
    )
    return hits / len(target_budgets)


def compute_scaling(
    compute_levels: Sequence[float],
    accuracies: Sequence[float],
) -> float:
    """
    Eq. 2 — Scaling: average slope của accuracy-vs-compute curve.

    Dùng linear regression slope (OLS) trên toàn bộ data points.
    Giá trị dương → accuracy tăng khi dùng nhiều compute.

    Args:
        compute_levels: Lượng compute tương đối (ví dụ: số lần append "Wait" = 0,1,2,4)
                        hoặc số thinking tokens (normalized)
        accuracies:     Accuracy tương ứng (0.0–1.0) ở mỗi compute level
    Returns:
        Slope (scaling score). Paper: BF = +15, RS = -35.

    Note: paper nhân với hệ số scale; ở đây return raw slope × 100 để align.
    """
    x = np.array(compute_levels, dtype=float)
    y = np.array(accuracies, dtype=float) * 100  # convert to percentage

    if len(x) < 2:
        return 0.0

    # OLS slope
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-9)
    return float(slope)


def compute_performance(accuracies: Sequence[float]) -> float:
    """
    Eq. 3 — Performance: maximum accuracy đạt được.

    Args:
        accuracies: List accuracy tại các compute levels
    Returns:
        Max accuracy (0.0–1.0)
    """
    return float(max(accuracies)) if accuracies else 0.0


def compute_all_metrics(
    compute_levels: Sequence[float],
    accuracies: Sequence[float],
    target_budgets: Sequence[int] | None = None,
    actual_tokens: Sequence[int] | None = None,
) -> ScalingMetrics:
    """
    Tính cả 3 metrics cùng lúc.

    Args:
        compute_levels:  Lượng compute (e.g., [0, 1, 2, 4] cho n_wait)
        accuracies:      Accuracy tương ứng (e.g., [0.50, 0.53, 0.57, 0.55])
        target_budgets:  (Optional) Cho Control metric
        actual_tokens:   (Optional) Cho Control metric
    """
    scaling = compute_scaling(compute_levels, accuracies)
    performance = compute_performance(accuracies)

    if target_budgets is not None and actual_tokens is not None:
        control = compute_control(target_budgets, actual_tokens)
    else:
        control = None

    return ScalingMetrics(control=control, scaling=scaling, performance=performance)


# ── Demo / Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Reproduce kết quả từ paper Table 3
    # BF: n_wait = [0, 1, 2, 4], accuracy trên AIME24
    bf_compute = [0, 1, 2, 4]
    bf_accuracy = [0.50, 0.53, 0.567, 0.533]  # approximate từ Figure 4

    metrics = compute_all_metrics(bf_compute, bf_accuracy)
    print("Budget Forcing:", metrics)

    # Rejection Sampling (RS) — negative scaling
    rs_compute = [1, 2, 4, 8]
    rs_accuracy = [0.40, 0.37, 0.33, 0.30]
    rs_metrics = compute_all_metrics(rs_compute, rs_accuracy)
    print("Rejection Sampling:", rs_metrics)
