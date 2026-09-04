import numpy as np
import pandas as pd
import pytest
from src.data.integrity import Layer1DataIntegrity
from src.features.pipeline import EconometricsFeaturePipeline


@pytest.fixture
def raw_market_data():
    dates = pd.date_range(start="2022-01-01", periods=150, freq="B")
    np.random.seed(42)
    base_price = 50.0 + np.cumsum(np.random.normal(0, 0.5, size=len(dates)))
    df = pd.DataFrame(
        {
            "open": base_price,
            "high": base_price + 1.0,
            "low": base_price - 1.0,
            "close": base_price + 0.2,
            "volume": np.random.randint(50000, 200000, size=len(dates)),
            "mkt_close": 1200.0 + np.cumsum(np.random.normal(0, 5, size=len(dates))),
            "mkt_return": np.random.normal(0, 0.01, size=len(dates)),
            "flat_microstructure_col": np.zeros(len(dates)),  # Cột rỗng cố tình đưa vào
        },
        index=dates,
    )
    return df


def test_zero_variance_purge(raw_market_data):
    df_audited, stats = Layer1DataIntegrity.run_audit(raw_market_data)
    # Cột có phương sai bằng 0 phải bị loại bỏ ngay lập tức
    assert "flat_microstructure_col" not in df_audited.columns
    assert "flat_microstructure_col" in stats["dropped_flat_cols"]


def test_pipeline_fit_transform(raw_market_data):
    df_audited, _ = Layer1DataIntegrity.run_audit(raw_market_data)
    df_audited["log_return"] = np.log(df_audited["close_adj"] / df_audited["close_adj"].shift(1))
    # Giả lập target label theo -1, 0, 1
    df_audited["target_label"] = np.random.choice([-1.0, 0.0, 1.0], size=len(df_audited))

    pipeline = EconometricsFeaturePipeline(target_horizon=5)
    pipeline.fit(df_audited)
    X_transformed = pipeline.transform(df_audited)

    # Đảm bảo pipeline đã fit và chọn được từ 6 đến 16 features theo thiết kế
    assert pipeline.is_fitted is True
    assert 6 <= len(pipeline.final_features_list_) <= 16
    assert X_transformed.shape[1] == len(pipeline.final_features_list_)
    # Giữ nguyên cấu trúc dòng index
    assert len(X_transformed) == len(df_audited)