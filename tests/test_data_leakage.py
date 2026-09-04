import numpy as np
import pandas as pd
import pytest
from src.models.validation import StandardPurgedCV
from src.data.labeling import Layer0TripleBarrier


@pytest.fixture
def dummy_timeseries_df():
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, size=len(dates))))
    df = pd.DataFrame(
        {
            "close": close,
            "close_adj": close,
            "open_adj": close * 0.99,
            "high_adj": close * 1.01,
            "low_adj": close * 0.98,
            "volume": np.random.randint(100000, 1000000, size=len(dates)),
            "log_return": np.insert(np.diff(np.log(close)), 0, 0.0),
        },
        index=dates,
    )
    return df


def test_purged_cv_boundary(dummy_timeseries_df):
    purge_gap = 5
    n_splits = 4
    cv = StandardPurgedCV(n_splits=n_splits, purge_gap=purge_gap)

    for train_idx, test_idx in cv.split(dummy_timeseries_df):
        max_train_idx = train_idx.max()
        min_test_idx = test_idx.min()
        # Kiểm tra khoảng cách an toàn: train kết thúc trước test ít nhất (purge_gap + 1)
        assert min_test_idx - max_train_idx > purge_gap, (
            f"Leakage detected: max_train={max_train_idx}, min_test={min_test_idx}, gap={purge_gap}"
        )
        # Không có phần tử trùng lặp giữa 2 tập
        assert len(set(train_idx).intersection(set(test_idx))) == 0


def test_triple_barrier_causality(dummy_timeseries_df):
    h = 5
    df_labeled = Layer0TripleBarrier.label(dummy_timeseries_df, h=h, pt=1.0, sl=1.0, vol_span=20)
    # Target của h dòng cuối cùng bắt buộc phải là NaN vì không đủ chân trời tính toán
    assert df_labeled["target_label"].iloc[-h:].isna().all()