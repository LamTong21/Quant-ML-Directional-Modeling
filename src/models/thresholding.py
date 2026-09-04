import numpy as np


def calculate_zero_anchored_thresholds(
    oof_probs: np.ndarray, min_margin: float = 0.04
) -> tuple[float, float]:
    """Tính toán ngưỡng Up Cut và Down Cut độc lập trên ma trận OOF."""
    oof_delta = oof_probs[:, 2] - oof_probs[:, 0]
    pos_deltas = oof_delta[oof_delta > 0.0]
    neg_deltas = oof_delta[oof_delta < 0.0]

    up_cut = (
        np.percentile(pos_deltas, 50) if len(pos_deltas) > 10 else min_margin
    )
    down_cut = (
        np.percentile(neg_deltas, 50) if len(neg_deltas) > 10 else -min_margin
    )

    up_cut = max(up_cut, min_margin)
    down_cut = min(down_cut, -min_margin)
    return float(up_cut), float(down_cut)