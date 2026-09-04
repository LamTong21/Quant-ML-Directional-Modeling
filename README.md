Dưới đây là nội dung chi tiết cho file **`README.md`**, được định hình liền mạch theo chuỗi nghiên cứu 3 giai đoạn (Three-Phase Quantitative Trajectory) tiếp nối 2 repo trước của bạn (`Quant-ML-Microstructure` và `Context-Aware-Quant-Feature-Engineering`), thể hiện đúng văn phong học thuật và kỹ thuật định lượng:

# From Microstructure to Macro-Alignment: Exogenous Context, Cross-Fold Consensus, and Leak-Free Directional Modeling in Emerging Equities

```
!Python 3.10+
!License: MIT
!Framework: Scikit--Learn / XGBoost / Optuna
```

---

## 📌 Research Trajectory & Context

Dự án này đánh dấu **Phase III** (Giai đoạn hoàn thiện pipeline dự báo hướng giá) trong chuỗi nghiên cứu có hệ thống về mô hình hóa chuỗi thời gian tài chính trên thị trường chứng khoán mới nổi (Emerging/Frontier Market - điển hình là HOSE, Việt Nam):

1. **Phase I: Quant-ML-Microstructure**
    - Thiết lập nền tảng toán học cho cấu trúc vi mô sổ lệnh nội phiên (L2 Limit Order Book), trích xuất các động lực học luồng lệnh thực tế: Order Flow Imbalance (OFI), Volume-Synchronized Probability of Toxicity (VPIN), và Micro-Price Dynamics.
2. **Phase II: Context-Aware-Quant-Feature-Engineering**
    - Mở rộng không gian trạng thái từ vi cấu trúc sang chuỗi động lực học kinh tế lượng đa quy mô: Fractional Differentiation ($d^*$), kiểm định bước nhảy Barndorff-Nielsen & Shephard Bipower Variation, Permutation Entropy (độ hỗn loạn vi mô), và phát hiện chế độ ngầm định (GMM / Markov Switching).
3. **Phase III (Repo này): Exogenous Alignment, Feature Consensus & Leak-Free Directional Inference**
    - Hợp nhất động lực học nội sinh (Endogenous) với bối cảnh thị trường vĩ mô ngoại sinh (Exogenous VN-Index Benchmark).
    - Chuẩn hóa phễu tinh lọc đặc trưng 4 cấp độ (Linear VIF, Non-linear HRP Spearman, Dual Granger/Transfer Entropy causality, và Directional Consensus Anchoring).
    - Triển khai quy trình kiểm định nghiêm ngặt chống rò rỉ thông tin (Leak-free Purged Nested Time-Series CV) kết hợp cân bằng trọng số mẫu nghịch đảo và tối ưu ngưỡng biên đối xứng qua mốc 0 (Zero-Anchored Symmetric Percentile Margin).
    - Kiểm nghiệm thực tế qua mô phỏng khớp lệnh trễ $T+1$ Mark-to-Market và chỉ ra **"Nghịch lý ma sát" (The Friction Paradox)**, đặt nền móng lý thuyết cho cấu trúc Meta-Labeling hai tầng.

---

## 🔬 System Architecture

Toàn bộ pipeline hoạt động khép kín qua 5 phân tầng kỹ thuật: 

```
[ Ingestion & Physical Audit ]
│
▼ (Forward-adjustment, Zero-variance sanitization)
[ Volatility-Scaled Triple Barrier ] ───> y ∈ {-1, 0, 1} (h = 5 bars)
│
▼
[ Stage 3: Exogenous & Directional Feature Space ] (78 - 105 Candidates)
│
▼
[ Stage 4: Consensus-Anchored Filtering ]
├── 1. Linear VIF Pruning (VIF < 6.0, |ρ| < 0.85)
├── 2. Non-linear HRP Clustering (Spearman Distance, K = 14)
├── 3. Causality Screening (Granger F-test p < 0.05 OR Transfer Entropy MI > 0.01)
└── 4. Dynamic Volatility-Scaled Lag Engine (6 ≤ |F_final| ≤ 16)
│
▼
[ Modeling & Inference ]
├── Purged Nested CV (K_outer = 5, K_inner = 3, Purge Gap = 5)
├── Inverse-Frequency Sample Weighting
├── Optuna Multi-Class Log-Loss Hyperparameter Tuning
└── Zero-Anchored Symmetric Percentile Thresholding (θ_up, θ_down)
│
▼
[ T+1 Realistic Mark-to-Market Simulation ] (20 bps friction, Vol-Target Sizing)
```

---

## 🚀 Key Theoretical & Empirical Contributions

### 1. Phễu làm sạch và ngăn ngừa ma trận kỳ dị (Ingestion Boundary)

- Triển khai forward-adjusted corporate actions để loại bỏ hoàn toàn lookahead bias.
- Cơ chế tự động quét và loại trừ biến có phương sai bằng 0 ($\text{Var}(X) < 10^{-8}$) trực tiếp tại cổng nạp, ngăn ngừa hiện tượng ma trận thiết kế không thể khả nghịch ($\det(X^TX) = 0$) trong các bài kiểm tra kinh tế lượng (VIF / Granger).

### 2. Tích hợp bối cảnh thị trường ngoại sinh (Exogenous Market Engine)

- Mở rộng thêm họ đặc trưng tương quan với chỉ số thị trường VN-Index:
    - **Relative Strength (RS):** Alpha đơn phiên ($r_t - r_{m,t}$) và Alpha tích lũy 20 phiên.
    - **Dynamic Rolling Beta:** Ước lượng độ nhạy thị trường trên cửa sổ lăn 60 phiên: $\beta_t = \frac{\text{Cov}_{60}(r, r_m)}{\text{Var}_{60}(r_m) + \epsilon}$.
    - **Relative Volatility Ratio:** Độ lệch rủi ro biến động giữa cổ phiếu và chỉ số chung.
    - **Signed Market Divergence:** Phân kỳ định hướng khi cổ phiếu đi ngược xu hướng thị trường: $\text{sign}(r_t)\cdot\text{sign}(r_{m,t})\cdot(r_t - r_{m,t})$.

### 3. Giải quyết hiện tượng lệch xác suất (Zero-Anchored Thresholding)

- Thay vì quét ngưỡng tĩnh hoặc tính percentile trên toàn bộ phân phối trôi dạt (concept drift), ngưỡng được tách rời độc lập trên hai miền:
    
    $$
    \Delta P_t = P(y_t = +1) - P(y_t = -1)
    $$
    
    $$
    \theta_{up} = \max(\mathcal{Q}_{0.50}(\Delta P^+), 0.04), \quad \theta_{down} = \min(\mathcal{Q}_{0.50}(\Delta P^-), -0.04)
    $$
    
- Cơ chế này loại bỏ hoàn toàn hiện tượng sụp đổ nhãn một chiều (one-sided prediction collapse), bảo toàn tính phân cực tự nhiên của thị trường.

### 4. Bằng chứng thực nghiệm về độ bền vững đặc trưng (Cross-Fold Consensus)

- Khảo sát 1,996 phiên giao dịch thực tế (2018–2026, 1,655 phiên OOS):
    - **Tính bền vững cao (>60% số Fold):** Độ biến thiên thanh khoản Amihud (`liq_amihud_z_acceleration`, `liq_amihud_z_lag2`) và động lượng đuôi béo Kurtosis (`kurtosis_20_momentum`) duy trì ưu thế xuyên suốt các chu kỳ thị trường khác nhau.
    - **Trọng số đóng góp lớn (Gain Top 10):** Biến phân kỳ thị trường ngoại sinh (`mkt_divergence_signed_acceleration`) và độ lệch phân phối Return (`skewness_20_zscaled_lag5`) vươn lên thành những nhân tố phân loại hướng giá mạnh mẽ nhất.

---

## 📊 Empirical Results

### Out-of-Sample Directional Performance ($N = 1,655$ bars)

| Metric / Class | Down (-1) | Neutral (0) | Up (+1) | Macro Avg | Weighted Avg |
| --- | --- | --- | --- | --- | --- |
| **Precision** | 0.4335 | 0.1192 | **0.5168** | 0.3565 | 0.4346 |
| **Recall** | **0.1470** | 0.6768 | **0.2000** | 0.3413 | 0.2350 |
| **$F_1$-Score** | 0.2196 | 0.2027 | **0.2884** | **0.2369** | 0.2496 |
- **Độ chính xác định hướng:** Tỷ lệ đoán trúng hướng vượt trội so với phân phối ngẫu nhiên 3 lớp, đặc biệt là chiều tăng giá (Precision: 51.68%).
- **Minority Capture:** Nhờ trọng số mẫu cân bằng (Inverse-Frequency), số lượng trường hợp bắt trúng rào cản chạm biên giảm tăng từ 82 lên 101, và biên tăng đạt 154 trường hợp.

---

## ⚖️ The Friction Paradox & Roadmap to Meta-Labeling

### Thực trạng mô phỏng Mark-to-Market $T+1$ (Phí giao dịch & Trượt giá = 20 bps, Vol-target = 15%)

- **Lợi nhuận lũy kế (Net):** -47.05%
- **Lợi nhuận hàng năm (Annualized):** -9.23%
- **Sharpe Ratio (Net):** -1.01
- **Max Drawdown (MDD):** -52.67%

### Giải mã mâu thuẫn giữa Accuracy và PnL

1. **Turnover Decay:** Việc tăng Recall hướng giá kéo theo tần suất giao dịch cao hơn. Trong điều kiện khớp lệnh trễ $T+1$, chi phí ma sát luân chuyển vị thế (Turnover friction) tích lũy bào mòn đường cong vốn.
2. **Mean-Reversion Latency:** Tín hiệu bứt phá do phân kỳ thị trường tại ngày $t$ thường gặp áp lực điều chỉnh kỹ thuật ngay tại phiên $t+1$ (thời điểm mở vị thế thực tế theo luật thị trường Việt Nam).
3. **Bất đối xứng vị thế:** Mô hình đơn tầng (Single-stage) đối xử cân xứng giữa Long (+1) và Short (-1), trong khi thị trường cơ sở bị chi phối bởi quy định cấm bán khống và chi phí vay mượn bất đối xứng.

### 🔮 Hướng phát triển tiếp theo (Phase IV):

Chuyển đổi toàn bộ hệ thống sang kiến trúc **Decoupled Two-Stage Meta-Labeling** (theo Marcos Lopez de Prado):

- **Stage 1 (Primary Model):** Kế thừa toàn bộ không gian đặc trưng của repo này để phát hiện tín hiệu định hướng thô (High-Recall Directional Signal).
- **Stage 2 (Secondary Meta-Model):** Mô hình phân loại nhị phân ($y_{\text{meta}} \in \{0, 1\}$) đóng vai trò bộ lọc thực thi và định cỡ vị thế (Position Sizing Filter), loại bỏ các cú bứt phá giả trước khi phát sinh chi phí giao dịch.

---

## 📂 Repository Structure

```
├── config/
│   └── default_config.yaml         # Cấu hình chuỗi thời gian, mã cổ phiếu, tham số barrier
├── data/                           # Thư mục lưu trữ dữ liệu thô và dữ liệu đã gán nhãn
├── notebooks/
│   └── Modeling05(Back_to_Feature_Engineering).ipynb # Pipeline mẫu hoàn chỉnh trên Colab
├── src/
│   ├── data/
│   │   ├── loaders.py              # Nạp dữ liệu cơ sở & VN-Index qua vnstock API
│   │   ├── integrity.py            # Kiểm toán hình học & loại bỏ Zero-variance
│   │   └── labeling.py             # Volatility-Scaled Triple Barrier Method
│   ├── features/
│   │   ├── diagnostics.py          # Kiểm định thống kê DGP (Hurst, VR, FracDiff, ARCH, Jumps)
│   │   ├── strategies/             # Các module trích xuất đặc trưng đa quy mô & thị trường
│   │   ├── selection.py            # Phễu lọc VIF, HRP Spearman, Granger Causality & MI
│   │   └── pipeline.py             # EconometricsFeaturePipeline tích hợp chuẩn Scikit-Learn
│   ├── models/
│   │   ├── validation.py           # Purged Time-Series Cross Validation engine
│   │   └── thresholding.py         # Zero-anchored percentile margin optimization
│   └── backtest/
│       ├── execution.py            # DirectionalMTMSimulator (Mô phỏng Mark-to-Market trễ T+1)
│       └── metrics.py              # Đánh giá Feature Importance (Gain) & Consensus
├── requirements.txt
└── README.md
```

## 🛠️ Quickstart

### 1. Cài đặt môi trường

```
git clone https://github.com/LamTong21/Emerging-Equities-Directional-ML.git
cd Emerging-Equities-Directional-ML
pip install -r requirements.txt
```

### 2. Thực thi pipeline hoàn chỉnh

```
from src.data.loaders import AssetAndMarketLoader
from src.data.integrity import Layer1DataIntegrity
from src.data.labeling import Layer0TripleBarrier
from src.features.pipeline import EconometricsFeaturePipeline

# 1. Nạp và đồng bộ dữ liệu cổ phiếu cơ sở & VN-Index
loader = AssetAndMarketLoader(symbol="DIG", benchmark_symbol="VNINDEX", start_date="2018-01-01", end_date="2026-06-01")
df_raw = loader.fetch_data()

# 2. Kiểm toán tính toàn vẹn & gán nhãn Triple Barrier
df_clean, _ = Layer1DataIntegrity.run_audit(df_raw)
df_labeled = Layer0TripleBarrier.label(df_clean, h=5, pt=1.0, sl=1.0, vol_span=20)

# 3. Fit Pipeline đặc trưng kinh tế lượng (chống rò rỉ thông tin)
pipeline = EconometricsFeaturePipeline(target_horizon=5)
pipeline.fit(df_labeled)
X_features = pipeline.transform(df_labeled)
```

## 📚 References

- **Amihud, Y. (2002).** *Illiquidity and stock returns: cross-section and time-series effects.* Journal of Financial Markets, 5(1), 31-56.
- **Barndorff-Nielsen, O. E., & Shephard, N. (2006).** *Econometrics of testing for jumps in financial economics using realized variance.* Journal of Financial Econometrics, 4(1), 1-30.
- **Corwin, S. A., & Schultz, P. (2012).** *A simple way to estimate bid-ask spreads from daily high and low prices.* The Journal of Finance, 67(2), 719-760.
- **De Prado, M. L. (2018).** *Advances in Financial Machine Learning.* John Wiley & Sons.
- **Engle, R. F., & Granger, C. W. (1987).** *Co-integration and error correction: Representation, estimation, and testing.* Econometrica, 55(2), 251-276.
- **Lo, A. W., & MacKinlay, A. C. (1988).** *Stock market prices do not follow random walks: Evidence from a simple specification test.* The Review of Financial Studies, 1(1), 41-66.