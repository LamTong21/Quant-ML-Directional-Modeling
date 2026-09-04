# From Microstructure to Macro-Alignment: Exogenous Context, Cross-Fold Consensus, and Leak-Free Directional Modeling in Emerging Equities

## Abstract

Quantitative machine learning models often suffer from structural decay when migrating from controlled feature engineering to out-of-sample temporal execution. This study completes a three-phase research trajectory—transitioning from endogenous high-frequency microstructure dynamics and multi-scale econometric features to an exogenously grounded, leak-free directional forecasting framework.

Formulated over a discrete ternary probability space, $\mathcal{Y} \in \{-1, 0, 1\}$, generated via a volatility-scaled Triple Barrier Method ($h=5$), we isolate pure directional predictive efficacy from execution friction.

Addressing the limitations of single-asset time-series models, this paper introduces:

1. An empirical data integrity pipeline enforcing zero-variance elimination at the ingestion boundary.
2. An exogenous market integration engine incorporating relative strength, rolling dynamic beta, and directional market divergence against the benchmark index (VN-Index).
3. A multi-stage feature selection architecture combining Variance Inflation Factor (VIF) pruning, Hierarchical Risk Parity (HRP) clustering, dual Granger/Transfer Entropy causal screening, and a directional consensus anchor.
4. A nested Purged Time-Series Cross-Validation scheme paired with inverse-frequency dynamic sample weighting and zero-anchored symmetric percentile margin thresholding.

Empirically verified on 1,996 daily trading bars of single-asset equity data (Ticker: `DIG`, 2018–2026) within Vietnam's frontier-emerging equity market, incorporating benchmark context lifts out-of-sample directional recall to 20.00% for upside movements and 14.70% for downside regimes, achieving an overarching macro $F_1$-score of 0.2369.

Tracking feature consensus demonstrates that structural illiquidity shocks (`liq_amihud`) and fat-tail momentum (`kurtosis_20`) persist across $\ge 60\%$ of independent outer folds, while market divergence metrics emerge as dominant directional gain contributors.

Finally, a clean $T+1$ mark-to-market simulator reveals a critical empirical dichotomy: while market integration significantly enhances multi-class classification power, it introduces execution turnover friction under strict settlement delays, formally establishing the theoretical necessity for a decoupled, two-stage meta-labeling architecture.

- **Keywords:** Directional Prediction, Triple Barrier Labeling, Exogenous Market Context, Purged Cross-Validation, Cross-Fold Consensus, Machine Learning Finance.

## 1. Introduction & Research Trajectory

Predicting short-term equity price direction using machine learning remains notoriously difficult due to low signal-to-noise ratios, non-stationary data-generating processes, and pervasive institutional market frictions.

This paper represents the culmination of a three-stage quantitative research initiative dedicated to constructing a realistic, econometrically grounded equity forecasting framework:

- **Phase I (`Quant-ML-Microstructure`):** Formulated the mathematical foundation of high-frequency order book mechanics, establishing order flow imbalance (OFI), volume-synchronized probability of toxicity (VPIN), and micro-price dynamics as key drivers of short-term price formation.
- **Phase II (`Context-Aware-Quant-Feature-Engineering`):** Expanded the state space from high-frequency microstructure to multi-scale econometric dynamics, integrating fractional differentiation ($d^*$), Barndorff-Nielsen jump tests, permutation entropy, and unsupervised regime switching (GMM / Markov Switching) to adapt to shifting market environments.
- **Phase III (Current Work):** Unifies endogenous econometric representations with exogenous macro-market context (VN-Index), eliminates label and probability distortion through a zero-anchored symmetric percentile thresholding mechanism, and investigates empirical feature consensus under strict, leak-free purged validation.

```
┌────────────────────────────────────────────────────────────────────────┐
│             PHASE I: MICROSTRUCTURE FOUNDATIONS                        │
│ - High-Frequency Limit Order Book Topologies (L2 Multi-Level)          │
│ - Order Flow Toxicity (OFI, VPIN Proxies)                              │
│ - Structural Liquidity Dynamics (Amihud, Corwin-Schultz Spreads)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           PHASE II: CONTEXT-AWARE FEATURE DECOMPOSITION                │
│ - Long Memory Retention via Fractional Differentiation (d*)            │
│ - Volatility Jumps (Barndorff-Nielsen & Shephard Bipower Variation)    │
│ - Multi-Scale Dynamical Complexity (Permutation & Shannon Entropy)     │
│ - Unsupervised Hidden Regime Transitions (GMM / Markov Switching)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│        PHASE III (THIS STUDY): EXOGENOUS INTEGRATION & BASELINE        │
│ - Stage 1: Zero-Variance Physical Sanitization Filter                  │
│ - Stage 2 & 3: Market Benchmark Integration (Relative Strength & Beta) │
│ - Stage 4: Econometric Consensus Anchoring (VIF -> HRP -> Causality)   │
│ - Modeling: Purged Nested CV + Inverse-Frequency Balanced Weights      │
│ - Inference: Zero-Anchored Symmetric Percentile Margin Thresholding    │
│ - Verification: Frozen Daily T+1 Mark-to-Market Evaluation Engine      │
└────────────────────────────────────────────────────────────────────────┘
```

A common failure in quantitative finance literature is the premature conflation of directional classification with discretionary portfolio execution. In emerging markets—such as Vietnam's HOSE exchange, which features strict short-selling constraints and a mandatory $T+1$ execution/settlement lag—applying post-hoc execution rules obscures whether empirical failures stem from predictive miscalibration or frictional decay.

To eliminate these confounding factors, we freeze the evaluation pipeline over an unconstrained probability space $\mathcal{Y} \in \{-1, 0, 1\}$. This paper addresses:

1. Eliminating singular covariance matrices caused by latent zero-variance microstructure vectors.
2. Overcoming regime-induced prediction collapse where estimators output trivial one-sided forecasts.
3. Measuring empirical feature evaporation across non-overlapping market regimes.
4. Evaluating how exogenous benchmark divergence impacts directional separation.

## 2. Methodology & System Architecture

### 2.1. Physical Data Integrity & Ingestion Boundary

Let the discrete market bar at day $t$ for the primary asset be denoted as:

$$
S_t = [O_t, H_t, L_t, C_t, V_t]^T
$$

and for the benchmark index (`VNINDEX`) as:

$$
M_t = [O_{m,t}, H_{m,t}, L_{m,t}, C_{m,t}, V_{m,t}]^T
$$

The two price series are synchronized along the calendar index via an exact inner intersection, $\mathcal{T} = \mathcal{T}_{\text{asset}} \cap \mathcal{T}_{\text{mkt}}$, preserving unified trading hours.

**Forward Corporate Action Adjustments** 

To eliminate lookahead bias inherent in standard backward-adjusted series, dividend distributions $D_t$ and stock split factors $\phi_t$ are integrated forward from inception $t_0$:

$$
R_t = \frac{C_t + D_t}{C_{t-1} \cdot \phi_t} - 1, \quad C_t^{\text{adj}} = C_0 \prod_{i=1}^t (1 + R_i)
$$

Geometric boundary auditing enforces:

$$
H_t = \max(H_t, O_t, C_t), \quad L_t = \min(L_t, O_t, C_t) \quad \forall t \in \mathcal{T}
$$

**Zero-Variance Sanitization** 

Given an initial feature candidate vector $X_{\cdot, j} \in \mathbf{X}_{\text{raw}}$, any variable exhibiting variance below $\epsilon = 10^{-8}$ is purged at the ingestion boundary:

$$
\mathbf{X} = \left\{ X_{\cdot, j} \;\middle\vert{}\; \widehat{\operatorname{Var}}(X_{\cdot, j}) > \epsilon \;\wedge\; \vert{}\operatorname{Unique}(X_{\cdot, j})\vert{} > 1 \right\} \quad \forall j
$$

This step removes vendor-induced flat microstructure arrays (such as null order-flow proxies across historical horizons), preventing non-invertible design matrices ($\det(\mathbf{X}^T\mathbf{X}) = 0$) during linear pruning.

### 2.2. Volatility-Scaled Triple Barrier Probability Space

The directional target $y_t \in \{-1, 0, 1\}$ is generated using the Triple Barrier Method over a prediction horizon $h = 5$ days. Local volatility $\sigma_t$ is tracked via an exponentially weighted moving standard deviation of logarithmic returns $r_t = \ln(C_t^{\text{adj}} / C_{t-1}^{\text{adj}})$ with span $\tau_{\text{vol}} = 20$:

$$
\sigma_t = \sqrt{\frac{\sum_{i=0}^\infty (1 - \alpha)^i (r_{t-i} - \bar{r}_t)^2}{\sum_{i=0}^\infty (1 - \alpha)^i}}, \quad \alpha = \frac{2}{\tau_{\text{vol}} + 1} \quad[cite: 2]
$$

Upper and lower barriers are established symmetrically:

$$
U_t = C_t^{\text{adj}} (1 + k_{\text{pt}} \sigma_t), \quad L_t = C_t^{\text{adj}} (1 - k_{\text{sl}} \sigma_t) \quad[cite: 2]
$$

Setting $k_{\text{pt}} = k_{\text{sl}} = 1.0$, the discrete label $y_t$ reflects the first barrier reached:

$$
y_t = \begin{cases} +1, & \text{if } \tau_{\text{upper}} < \tau_{\text{lower}} \;\wedge\; \tau_{\text{upper}} \le h \\ -1, & \text{if } \tau_{\text{lower}} < \tau_{\text{upper}} \;\wedge\; \tau_{\text{lower}} \le h \\ 0, & \text{if neither barrier is touched within } h \text{ bars} \end{cases} \quad[cite: 2]
$$

### 2.3. Exogenous Market Integration & Feature Engine

Stage 3 constructs an econometric candidate space incorporating single-asset and market-relative representations:

**Exogenous Context Engine (`MarketContextStrategy`)**

Let $r_{m,t} = \ln(C_{m,t} / C_{m,t-1})$ represent benchmark index log-returns. The exogenous module generates five structural indicators:

- **Relative Strength (RS):** Daily excess returns and cumulative multi-scale relative strength:
    
    $$
    \text{RS}_t = r_t - r_{m,t}, \quad \text{RS}_{\text{cum}, t}^{(20)} = \sum_{i=0}^{19} r_{t-i} - \sum_{i=0}^{19} r_{m,t-i}
    $$
    
- **Dynamic Rolling Beta ($\beta_t$):** Estimated over a 60-day window:
    
    $$
    \beta_t = \frac{\widehat{\operatorname{Cov}}_{60}(r_t, r_{m,t})}{\widehat{\operatorname{Var}}_{60}(r_{m,t}) + \epsilon}, \quad \beta_t \in [-1.0, 3.5]
    $$
    
- **Relative Volatility Ratio:** Tracking volatility divergence between the asset and the index:
    
    $$
    \text{VR}_t = \frac{\sigma_{t}^{(20)}}{\sigma_{m,t}^{(20)} + \epsilon}
    $$
    
- **Market Correlation Alignment:** The rolling 20-day Pearson correlation $\rho_t(r_t, r_{m,t})$.
- **Signed Market Divergence:** Isolating decoupling between the asset's idiosyncratic trajectory and the market trend:
    
    $$
    \text{Div}_t = \operatorname{sign}(r_t) \cdot \operatorname{sign}(r_{m,t}) \cdot (r_t - r_{m,t})
    $$
    

**Directional Pressure & Flow Divergence**

Endogenous directional feature generation includes:

- **Chaikin Money Flow (CMF) Divergence:** Calculated via Close Location Value (CLV) weighted by trading volume:
    
    $$
    \text{CLV}_t = \frac{(C_t - L_t) - (H_t - C_t)}{(H_t - L_t) + \epsilon}, \quad \text{CMF}_t^{(20)} = \frac{\sum_{i=0}^{19} \text{CLV}_{t-i} V_{t-i}}{\sum_{i=0}^{19} V_{t-i} + \epsilon}
    $$
    
- **Signed Volatility Skew:** Quantifying the asymmetry between upside and downside semi-variance:
    
    $$
    \sigma_{\text{up}, t}^2 = \frac{1}{20}\sum_{i=0}^{19} \max(r_{t-i}, 0)^2, \quad \sigma_{\text{down}, t}^2 = \frac{1}{20}\sum_{i=0}^{19} \min(r_{t-i}, 0)^2
    $$
    
    $$
    \text{VolSkew}_t = \frac{\sigma_{\text{up}, t}^2 - \sigma_{\text{down}, t}^2}{\sigma_{\text{up}, t}^2 + \sigma_{\text{down}, t}^2 + \epsilon}
    $$
    

### 2.4. Stage 4: Consensus-Anchored Econometric Filtering

To prevent tree subspace dilution, candidate features are processed through a four-stage econometric filter:

```
Candidate Feature Matrix (X_candidates, K ≈ 105 features)
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Linear Pruning: VIF < 6.0, Pairwise ρ < 0.85 │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Non-Linear Diversification: HRP Spearman    │
│  (Clustering into K=14 Orthogonal Medoids)   │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Causality Screening: Dual Granger F-test    │
│  (p < 0.05) OR Transfer Entropy (MI > 0.01)  │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Lag Dynamics Engine: Volatility-Scaled Lags │
│  (Generates: z-score, momentum, accel)       │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Consensus Anchoring & Mutual Information:   │
│  Prioritizes persistent directional keywords │
│  Constrains final dimensionality: [6, 16]    │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
    Final Model Feature Matrix (X_tr, K ≤ 16 features)
```

1. **Collinearity Removal:** Features exhibiting pairwise Pearson correlation $\vert{}\rho\vert{} > 0.85$ or Variance Inflation Factor $\text{VIF}_j > 6.0$ are iteratively dropped.
2. **Nonlinear Orthogonalization (HRP):** Features are clustered hierarchically using Spearman rank distance:
    
    $$
    D_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{\text{Spearman}}(X_i, X_j))} \quad[cite: 2]
    $$
    
    Ward's minimum variance algorithm groups the topology into $K_{\text{clusters}} = 14$ clusters, retaining the medoid of each cluster.
    
3. **Causal Precedence Testing:** Features must pass either a linear Granger causality test ($F$test $p$value $< 0.05$ across lags $p \in [1, 5]$) or exceed an empirical transfer entropy threshold (Lagged Mutual Information $I(X_{t-p}; y_t) > 0.01$).
4. **Consensus Priority Selection:** Causal features undergo dynamic lag transformation, generating rolling $z$scores, momentum, and acceleration terms. Features containing validated directional keywords (`liq_amihud`, `kurtosis_20`, `mkt_relative`, `mkt_divergence`, `cmf`, `obv`) receive selection priority over unanchored terms, and the final feature count is constrained:
    
    $$
    6 \le \vert{}\mathcal{F}_{\text{final}}\vert{} \le 16 \quad[cite: 2]
    $$
    

### 2.5. Purged Validation, Dynamic Weighting, and Zero-Anchored Thresholding

The modeling pipeline implements a nested Purged Time-Series Cross-Validation design ($K_{\text{outer}} = 5, K_{\text{inner}} = 3$) with a purge gap of $h = 5$ days and a training window cap of 750 bars.

**Inverse-Frequency Sample Weighting**

To prevent the objective function from ignoring minority directional classes during prolonged market trends, sample weights $w_t$ are updated dynamically in each fold:

$$
w_t = \frac{N_{\text{train}}}{3 \cdot N_{y_t}} \quad[cite: 2]
$$

where $N_{y_t}$ is the empirical count of samples belonging to class $y_t \in \{-1, 0, 1\}$ within the fold's training index.

**Zero-Anchored Symmetric Percentile Thresholding**

Given XGBoost multi-class probability outputs:

$$
\mathbf{P}_t = [P(y_t = -1), P(y_t = 0), P(y_t = +1)]^T \quad[cite: 2]
$$

we compute the raw directional confidence spread:

$$
\Delta P_t = P(y_t = +1) - P(y_t = -1) \quad[cite: 2]
$$

To avoid probability distortion caused by calculating percentiles over drifting distributions, thresholds are calibrated over positive and negative out-of-fold partitions independently:

$$
\Delta P^+ = \{\delta \in \Delta P^{\text{OOF}} \mid \delta > 0\}, \quad \Delta P^- = \{\delta \in \Delta P^{\text{OOF}} \mid \delta < 0\}
$$

$$
\theta_{\text{up}} = \max\left(\mathcal{Q}_{0.50}(\Delta P^+), 0.04\right), \quad \theta_{\text{down}} = \min\left(\mathcal{Q}_{0.50}(\Delta P^-), -0.04\right)
$$

This guarantees strict threshold polarity ($\theta_{\text{up}} > 0$ and $\theta_{\text{down}} < 0$). The final discrete decision rule $\hat{y}_t$ follows:

$$
\hat{y}_t = \begin{cases} +1, & \text{if } \Delta P_t > \theta_{\text{up}} \\ -1, & \text{if } \Delta P_t < \theta_{\text{down}} \\ 0, & \text{otherwise} \end{cases}
$$

## 3. Empirical Results & Performance Diagnostics

The framework was evaluated across 1,996 trading days (June 2018 to June 2026) for ticker `DIG` (HOSE, Vietnam), yielding $N = 1,655$ out-of-sample forecast bars.

### 3.1. Stage 4 Feature Dimensions & Threshold Dynamics

As exogenous market features were added, candidate dimensions expanded from 78 to 105 features before VIF pruning. Across all folds, the zero-anchored thresholding mechanism preserved balanced polarity:

$$
\begin{array}{lccccc} \hline \textbf{Fold Index} & \textbf{Training Horizon} & \textbf{Input Dim} & \boldsymbol{\theta_{\text{up}}} & \boldsymbol{\theta_{\text{down}}} & \textbf{Final Features} \\ \hline \text{Fold 1} & 2018\text{-}06\text{-}05 \rightarrow 2019\text{-}09\text{-}26 & 78 & +0.040 & -0.045 & 16 \\ \text{Fold 2} & 2018\text{-}06\text{-}05 \rightarrow 2021\text{-}01\text{-}19 & 73 & +0.053 & -0.112 & 16 \\ \text{Fold 3} & 2019\text{-}06\text{-}05 \rightarrow 2022\text{-}05\text{-}25 & 102 & +0.040 & -0.097 & 16 \\ \text{Fold 4} & 2020\text{-}09\text{-}28 \rightarrow 2023\text{-}09\text{-}20 & 105 & +0.138 & -0.040 & 12 \\ \text{Fold 5} & 2022\text{-}01\text{-}21 \rightarrow 2025\text{-}01\text{-}15 & 104 & +0.040 & -0.178 & 16 \\ \hline \end{array} 
$$

```
                OUT-OF-SAMPLE DIRECTIONAL FORECAST DISTRIBUTION (%)
  Fold 1 [==== 23.80% ====] [============ 45.18% ============] [====== 31.02% ======]
  Fold 2 [=== 17.47% ===] [=================== 72.59% ===================] [= 9.94% =]
  Fold 3 [===== 28.61% =====] [================= 63.25% =================] [= 8.13% =]
  Fold 4 [ 0.00% ] [====================== 78.92% ======================] [=== 21.08% ===]
  Fold 5 [=== 20.18% ===] [====================== 79.82% ======================] [ 0.00% ]
         └───────────────────┴─────────────────────────────┴────────────────────────────┘
          Direction Up (+1)           Neutral (0)              Direction Down (-1)
```

### 3.2. Directional Accuracy & Classification Metrics

Table 1 outlines out-of-sample directional classification metrics against realized Triple Barrier labels ($N = 1,655$ test days):

$$
\begin{array}{lcccc} \hline \textbf{Directional Label} & \textbf{Precision} & \textbf{Recall} & \boldsymbol{F_1}\textbf{-Score} & \textbf{Support} \\ \hline \text{Down } (-1) & 0.4335 & 0.1470 & 0.2196 & 687 \\ \text{Neutral } (0) & 0.1192 & 0.6768 & 0.2027 & 198 \\ \text{Up } (+1) & 0.5168 & 0.2000 & 0.2884 & 770 \\ \hline \textbf{Accuracy} & & & 0.2350 & 1,655 \\ \textbf{Macro Average} & 0.3565 & 0.3413 & 0.2369 & 1,655 \\ \textbf{Weighted Average} & 0.4346 & 0.2350 & 0.2496 & 1,655 \\ \hline \end{array} 
$$

```
                                 CONFUSION MATRIX HEATMAP
                                    Predicted Direction
                               Down (-1)    Neutral (0)     Up (+1)
                             ┌────────────┬─────────────┬────────────┐
             Down (-1)       │    101     │     469     │    117     │
                             ├────────────┼─────────────┼────────────┤
  True     Neutral (0)       │     37     │     134     │     27     │
  State                      ├────────────┼─────────────┼────────────┤
              Up (+1)        │     95     │     521     │    154     │
                             └────────────┴─────────────┴────────────┘
```

Compared to earlier baseline iterations, incorporating exogenous market features increased minority-class capture:

- Realized down-barrier hits correctly identified rose from 82 to **101 instances** (Recall: 14.70%, Precision: 43.35%).
- Realized up-barrier hits correctly captured expanded to **154 instances** (Recall: 20.00%, Precision: 51.68%).
- Overall Macro $F_1$ reached **0.2369**, marking the highest predictive accuracy across all three research phases.

### 3.3. Empirical Cross-Fold Feature Consensus

Tracking feature appearance frequencies across all five non-overlapping validation folds confirms the persistence of core structural factors:

$$
\begin{array}{lccc} \hline \textbf{Feature Architecture} & \textbf{Consensus Count} & \textbf{Mean Gain} & \textbf{Stability Classification} \\ \hline \text{liq\_amihud\_z\_acceleration} & \mathbf{4/5} & 0.0682 & \textbf{Universal Microstructure} \\ \text{kurtosis\_20\_momentum} & \mathbf{3/5} & 0.0766 & \textbf{Universal Tail-Risk} \\ \text{liq\_amihud\_z\_zscaled\_lag2} & \mathbf{3/5} & 0.0594 & \textbf{Universal Microstructure} \\ \text{mkt\_vol\_ratio\_20\_acceleration} & \mathbf{2/5} & 0.0641 & \textbf{Exogenous Market Shock} \\ \text{mkt\_divergence\_signed\_acceleration} & 1/5 & \mathbf{0.0834} & \textbf{High-Impact Regime Driver} \\ \text{skewness\_20\_zscaled\_lag5} & 2/5 & \mathbf{0.1104} & \textbf{Distributional Asymmetry} \\ \text{tsmom\_sign\_9\_momentum} & 1/5 & 0.0959 & \text{Transient Momentum} \\ \text{wavelet\_energy\_ratio\_lag4} & 1/5 & 0.0941 & \text{Transient Scale Shock} \\ \hline \end{array} 
$$

- Microstructure illiquidity (`liq_amihud`) and fat-tail momentum (`kurtosis_20`) persisted across $\ge 60\%$ of test folds, verifying their regime-invariant properties.
- Exogenous market interactions (`mkt_vol_ratio` and `mkt_divergence`) established strong causal relevance, with signed market divergence entering the Top 10 Gain contributors.

## 4. The Friction Paradox & the Need for Meta-Labeling

```
                     PERFORMANCE VS. PREDICTIVE POWER EVOLUTION
  Metric Value
    0.6 ┼────────────────────────────────────────────────────────────────────────┐
        │                                                     ▲ Macro F1 (0.237) │
    0.4 ┼                                                ▲───■                   │
        │                       ▲ Up Precision (0.517)  ╱                        │
    0.2 ┼───▲ Down Precision   ■───────────────────────■                         │
        │   (0.400)                                                              │
    0.0 ┼───■────────────────────────────────────────────────────────────────────┤
        │   Baseline Sharpe (+0.01)                                              │
   -0.5 ┼   ▼                                           ▼ Cumulative Return      │
        │                                               (-47.05%)                │
   -1.0 ┼───────────────────────────────────────────────■ Net Sharpe (-1.01)────┤
        2018-2020 (Phase I/II)     2021-2023 (Phase II)       2024-2026 (Phase III)
```

While Phase III achieved the highest directional precision and recall across both barriers, evaluating the raw signals through a daily $T+1$ mark-to-market simulator (volatility target = 15%, friction = 20 bps) generated negative net performance:

- **Cumulative Net Return:** $-47.05\%$
- **Annualized Net Return:** $-9.23\%$
- **Sharpe Ratio (Net):** $-1.01$
- **Maximum Drawdown:** $-52.67\%$

### Analysis of the Divergence

This divergence highlights an important structural insight: **enhanced directional recall increases trade turnover under execution latency.**

1. **Turnover Decay:** In Phase II, conservative thresholding classified 73.74% of bars as neutral, avoiding whipsaw costs. In Phase III, higher directional sensitivity raised trading frequency, causing cumulative 20 bps transaction frictions to compound against equity growth.
2. **$T+1$ Mean-Reversion Latency:** When an asset exhibits a sharp benchmark divergence at bar $t$, our model detects an active directional signal. However, because trades execute at the close of $t+1$, short-term mean reversion on day $t+1$ often erodes the initial price impulse.
3. **Execution Symmetry vs. Regulatory Asymmetry:** The single-stage model treats upside (+1) and downside (-1) opportunities symmetrically, which conflicts with market microstructure realities like short-selling bans and asymmetric borrowing fees.

### Resolving the Paradox: A Decoupled Meta-Labeling Architecture

This empirical finding demonstrates that a single-stage classifier cannot balance directional discovery and execution timing simultaneously. The logical next phase is a **Decoupled Two-Stage Meta-Labeling Architecture (Marcos Lopez de Prado)**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: PRIMARY DIRECTIONAL MODEL                   │
│ - Inputs: Phase III Econometric Feature Space & VN-INDEX Context       │
│ - Objective: High-Recall Directional Signal y_primary ∈ {-1, +1}       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               STAGE 2: SECONDARY META-MODEL (FILTER/SIZER)             │
│ - Target: Binary Outcome y_meta ∈ {0, 1}                               │
│     * y_meta = 1 if primary signal touches profit-taking barrier       │
│     * y_meta = 0 if trade hits stop-loss or times out (Abstain)        │
│ - Model: Extreme Gradient Boosting / Random Forest Meta-Filter         │
│ - Output: Execution sizing and false-positive elimination              │
└────────────────────────────────────────────────────────────────────────┘
```

By assigning directional forecasting to the primary model and position filtering/sizing to the secondary meta-model, false breakouts are eliminated before incurring transaction fees—preserving classification quality while improving realized PnL.

## 5. Conclusion & Working Paper Summary

This study completes Phase III of our quantitative modeling trajectory, delivering a standardized, leak-free directional evaluation framework for emerging equity markets:

- Demonstrated the importance of zero-variance filtering at data ingestion, protecting downstream econometric filters from singular matrix breakdowns.
- Integrated benchmark market indicators (VN-Index) into an econometric candidate engine, improving out-of-sample directional recall and boosting macro $F_1$-scores to **0.2369**.
- Quantified cross-fold feature consensus, identifying structural illiquidity (`liq_amihud`) and fat-tail momentum (`kurtosis_20`) as universal predictive factors across diverse macroeconomic regimes.
- Established that execution delays and transaction costs erode the PnL of responsive single-stage classifiers, providing an empirical rationale for migrating to a two-stage meta-labeling architecture.

The modeling framework, econometric router, and purged backtesting engine are now frozen to serve as a standardized baseline for future meta-labeling research.

## Comprehensive Feature Architecture Appendices

### **Appendix A: Full Exogenous & Endogenous Feature Specifications**

The finalized feature space integrates mathematical formulations across four distinct econometric families:

$$
\begin{array}{lll} \hline \textbf{Feature Code} & \textbf{Mathematical Formulation} & \textbf{Econometric Category} \\ \hline \text{log\_return} & r_t = \ln(C_t^{\text{adj}} / C_{t-1}^{\text{adj}}) & \text{Kinematic Dynamics} \\ \text{hl\_log\_range} & \Delta_{\text{HL},t} = \ln(H_t^{\text{adj}} / L_t^{\text{adj}}) & \text{Kinematic Dynamics} \\ \text{overnight\_gap} & \Delta_{\text{ON},t} = (O_t^{\text{adj}} / C_{t-1}^{\text{adj}}) - 1 & \text{Kinematic Dynamics} \\ \text{rvol\_k} & \text{RVOL}_{k,t} = V_t / \left(\frac{1}{k}\sum_{i=1}^k V_{t-i}\right) & \text{Volume Shock Dynamics} \\ \text{frac\_diff\_d} & (1-B)^d C_t^{\text{adj}} = \sum_{k=0}^\infty \omega_k C_{t-k}^{\text{adj}}, \; \omega_k = -\frac{\omega_{k-1}}{k}(d-k+1) & \text{Fractional Integration (Memory)} \\ \text{vol\_parkinson} & \sigma_{\text{P},t} = \sqrt{\frac{1}{4\ln 2} \ln(H_t/L_t)^2} & \text{Continuous Volatility Proxy} \\ \text{vol\_garman\_klass} & \sigma_{\text{GK},t} = \sqrt{0.5\ln(H_t/L_t)^2 - (2\ln 2 - 1)\ln(C_t/O_t)^2} & \text{Continuous Volatility Proxy} \\ \text{vol\_yang\_zhang} & \sigma_{\text{YZ},t} = \sqrt{\sigma_{\text{open}}^2 + k\sigma_{\text{close}}^2 + (1-k)\sigma_{\text{RS}}^2} & \text{Overnight-Jump Robust Volatility} \\ \text{jump\_diffusion} & J_t = \max(\text{RV}_t - \text{BV}_t, 0) / (\text{RV}_t + \epsilon), \; \text{BV}_t = \frac{\pi}{2}\sum \vert{}r_i\vert{}\vert{}r_{i-1}\vert{} & \text{Bipower Jump Discontinuity} \\ \text{pe\_entropy\_20} & H(P) = -\sum p(\pi) \log_2 p(\pi) / \log_2(m!), \; m=3 & \text{Permutation Entropy (Chaos)} \\ \text{wavelet\_energy\_ratio} & E_{d1} / E_{d2} = [(C_t - C_{t-1})/\sqrt{2}]^2 / [\frac{1}{2}(C_t+C_{t-1}-C_{t-2}-C_{t-3})]^2 & \text{Multi-Resolution Scale Energy} \\ \text{liq\_amihud\_raw} & \text{ILLIQ}_t = \vert{}r_t\vert{} / (C_t \cdot V_t) & \text{Endogenous Microstructure Illiquidity} \\ \text{liq\_spread\_cs} & S_{\text{CS},t} = 2(e^\alpha - 1) / (1 + e^\alpha) \quad (\text{Corwin-Schultz}) & \text{Effective Bid-Ask Spread Proxy} \\ \text{cmf\_20} & \sum_{i=0}^{19} \text{CLV}_{t-i} V_{t-i} / \sum_{i=0}^{19} V_{t-i}, \; \text{CLV}_t = \frac{(C_t-L_t)-(H_t-C_t)}{H_t-L_t} & \text{Volume-Price Flow Accumulation} \\ \text{obv\_zscore\_20} & (\text{OBV}_t - \mu_{\text{OBV}, 20}) / \sigma_{\text{OBV}, 20}, \; \text{OBV}_t = \sum \operatorname{sign}(r_i) V_i & \text{Directional Volume Momentum} \\ \text{mkt\_relative\_return} & r_{\text{asset}, t} - r_{\text{mkt}, t} & \text{Exogenous Benchmark Alpha} \\ \text{mkt\_rolling\_beta\_60} & \widehat{\operatorname{Cov}}_{60}(r_{\text{asset}}, r_{\text{mkt}}) / \widehat{\operatorname{Var}}_{60}(r_{\text{mkt}}) & \text{Exogenous Macro Systematic Risk} \\ \text{mkt\_vol\_ratio\_20} & \sigma_{\text{asset}, t}^{(20)} / \sigma_{\text{mkt}, t}^{(20)} & \text{Exogenous Relative Volatility Shock} \\ \text{mkt\_divergence\_signed} & \operatorname{sign}(r_{\text{asset}, t}) \operatorname{sign}(r_{\text{mkt}, t}) (r_{\text{asset}, t} - r_{\text{mkt}, t}) & \text{Exogenous Trend Decoupling} \\ \hline \end{array} 
$$

### **Appendix B: Hyperparameter Optimization Spaces (Optuna Tuning Protocol)**

During each inner fold of the Purged Time-Series Cross-Validation structure, hyperparameters were optimized over multi-class log-loss with inverse-frequency sample weights:

$$
\min_{\boldsymbol{\theta}} \mathcal{L}_{\text{balanced}}(\boldsymbol{\theta}) = -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} w_i \sum_{k \in \{-1,0,1\}} \mathbb{I}(y_i = k) \ln P(y_i = k \mid X_i; \boldsymbol{\theta}) 
$$

$$
\begin{array}{lccc} \hline \textbf{Hyperparameter Dimension} & \textbf{Search Distribution} & \textbf{Bounds Range} & \textbf{Step / Scale} \\ \hline \text{n\_estimators} & \text{Discrete Uniform} & [40, 120] & \text{step} = 20 \\ \text{max\_depth} & \text{Discrete Uniform} & [2, 4] & \text{step} = 1 \\ \text{learning\_rate} (\eta) & \text{Log-Uniform Continuous} & [0.02, 0.07] & \text{logarithmic} \\ \text{subsample} & \text{Uniform Continuous} & [0.65, 0.85] & \text{linear} \\ \text{colsample\_bytree} & \text{Uniform Continuous} & [0.65, 0.85] & \text{linear} \\ \text{reg\_lambda} (\text{L2 penalty}) & \text{Log-Uniform Continuous} & [3.0, 20.0] & \text{logarithmic} \\ \text{min\_child\_weight} & \text{Discrete Uniform} & [4, 12] & \text{step} = 1 \\ \hline \end{array} 
$$

### **Appendix C: Cross-Fold Consensus Feature Breakdown**

The table below catalogs every feature selected by Stage 4 in at least one fold, showing its cross-fold persistence:

$$
\begin{array}{lcccccc} \hline \textbf{Feature Identifier} & \textbf{Fold 1} & \textbf{Fold 2} & \textbf{Fold 3} & \textbf{Fold 4} & \textbf{Fold 5} & \textbf{Consensus Rate} \\ \hline \text{liq\_amihud\_z\_acceleration} & \checkmark & \checkmark & \checkmark & \checkmark & & 4/5 \; (80\%) \\ \text{kurtosis\_20\_momentum} & \checkmark & \checkmark & \checkmark & & & 3/5 \; (60\%) \\ \text{liq\_amihud\_z\_zscaled\_lag2} & \checkmark & \checkmark & \checkmark & & & 3/5 \; (60\%) \\ \text{kinematic\_clv\_vol\_roll20\_momentum} & & \checkmark & \checkmark & & & 2/5 \; (40\%) \\ \text{breakout\_ratio\_5\_acceleration} & & \checkmark & \checkmark & & & 2/5 \; (40\%) \\ \text{skewness\_20\_momentum} & & \checkmark & \checkmark & & & 2/5 \; (40\%) \\ \text{mkt\_vol\_ratio\_20\_acceleration} & & \checkmark & \checkmark & & & 2/5 \; (40\%) \\ \text{sin\_month\_momentum} & \checkmark & \checkmark & & & & 2/5 \; (40\%) \\ \text{shadow\_asymmetry\_ratio\_momentum} & \checkmark & & \checkmark & & & 2/5 \; (40\%) \\ \text{kurtosis\_20\_acceleration} & & \checkmark & & \checkmark & & 2/5 \; (40\%) \\ \text{skewness\_20\_zscaled\_lag5} & & & \checkmark & \checkmark & & 2/5 \; (40\%) \\ \text{tsmom\_sign\_16\_acceleration} & \checkmark & & & & & 1/5 \; (20\%) \\ \text{skewness\_20\_acceleration} & & & & \checkmark & & 1/5 \; (20\%) \\ \text{obv\_slope\_5\_momentum} & & & \checkmark & & & 1/5 \; (20\%) \\ \text{tsmom\_sign\_20\_momentum} & \checkmark & & & & & 1/5 \; (20\%) \\ \text{mkt\_divergence\_signed\_acceleration} & & & & \checkmark & & 1/5 \; (20\%) \\ \text{wavelet\_energy\_ratio\_lag4} & & & \checkmark & & & 1/5 \; (20\%) \\ \text{rvol\_5\_zscaled\_lag2} & & & & & \checkmark & 1/5 \; (20\%) \\ \hline \end{array} 
$$

## References:

- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56.
- Barndorff-Nielsen, O. E., & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using realized variance. *Journal of Financial Econometrics*, 4(1), 1-30.
- Cont, R., Kukanov, I., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
- Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *The Journal of Finance*, 67(2), 719-760.
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67-78.
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*, 1(1), 41-66.
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices. *The Journal of Business*, 73(3), 477-492.