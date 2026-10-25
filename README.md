# 72-Hours
# Cascade72: Urban Cascading Failure Forecasting Engine (72-Hour Horizon)

Cascade72 is a real-time predictive intelligence system that forecasts cascading failures across urban infrastructure networks up to 72 hours ahead. 

Engineered strictly in **Python 3.11**, using **NumPy** and **pandas** exclusively (no external machine learning or deep learning libraries), Cascade72 reconstructs delayed, out-of-order telemetry streams, identifies and downweights Byzantine or corrupted sensors, uncovers directed causal cascade networks via Time-Lagged Cross-Correlation (TLCC), computes 4-tier Bayesian risk probabilities, projects time-to-critical horizons, and synthesizes high-leverage emergency interventions.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw Event Stream (CSV)                |
                      | (Delayed, Out-of-Order Telemetry)     |
                      +-------------------+-------------------+
                                          |
                                          v  [ arrival_time <= now ]
                      +-------------------+-------------------+
                      | Source Reliability Filter             |
                      | - Temporal Median Consensus           |
                      | - Dynamic Gradient (dv/dt) Check      |
                      | - Anti-Stuck Variance Test            |
                      | - Adaptive Bayesian Updating          |
                      +-------------------+-------------------+
                                          | Reliable Weights w_s
                                          v
                      +-------------------+-------------------+
                      | Time-Series Reconstruction            |
                      | - Nadaraya-Watson Kernel Smoothing    |
                      | - Discretized Regularized Grid (dt)   |
                      +-------------------+-------------------+
                                          | Node Stress x_u(t)
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+--------------------+-------------------+   +-----------------+--------------------+
| Causal Cascade Graph Discovery         |   | State & Probability Estimator        |
| - Innovation Differencing Δx(t)        |   | - Zone Weighted Aggregate Stress     |
| - Directional TLCC Cross-Correlation   |   | - Stress Momentum dS/dt              |
| - Physical Domain Topology Gating      |   | - Incoming Cascade Pressure          |
| - Fisher z-Transformation Confidence   |   | - 4-State Softmax Simplex            |
| - Transitive Graph Reduction           |   | - Categorical Zone & City State      |
+--------------------+-------------------+   +-----------------+--------------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      | Forward Cascade Forecaster            |
                      | - Coupled Network ODE Simulation      |
                      | - Time-to-Critical (TTC) Prediction   |
                      | - Backward Causal Chain Extraction    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      | Intervention Engine                   |
                      | - Bottleneck Network Centrality       |
                      | - Domain-Specific Action Generation   |
                      | - Priority & Risk Reduction Ranking   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      | ENGINE OUTPUT (output/engine_output)  |
                      | (now, city_state, zones, edges, ...)  |
                      +---------------------------------------+
```

---

## Mathematical Formulations

### 1. Robust Source Reliability Estimation
Telemetry sources $s$ may be corrupted, Byzantine (inverted), noisy, or mechanically frozen (stuck). The filter evaluates three independent signals without prior knowledge of ground truth:

1. **Temporal Median Consensus Discrepancy**:
   Observations for node $(z, m)$ are matched against the local peer median $m_{(z, m)}(t)$. Residuals $e_i = |v_i - m_{(z, m)}(t_i)|$ yield composite error:
   $$E_s = 0.40 \cdot \text{Mean}(e_i) + 0.60 \cdot \text{Median}(e_i)$$
   $$S_{\text{consensus}}(s) = \exp\left( - \left(\frac{E_s}{\sigma_{\text{consensus}}}\right)^2 \right), \quad \sigma_{\text{consensus}} = 0.18$$

2. **Physical Rate Gradient Invariance**:
   Physical urban variables obey maximum rate limits $|dv/dt| \le R_{\max}$ (0.70 stress units/hour). Unphysical step jumps incur an exponential penalty:
   $$S_{\text{rate}}(s) = \exp\left( - 2.0 \cdot \text{Mean}\left( \max(0, |\Delta v/\Delta t| - R_{\max}) \right) \right)$$

3. **Anti-Stuck Sensor Test**:
   Sensors with $N \ge 8$ samples and zero variance $\text{Var}(v_s) < 10^{-4}$ are identified as frozen:
   $$S_{\text{stuck}}(s) = 0.05 \quad \text{if stuck, else } 1.0$$

4. **Sample-Adaptive Bayesian Update**:
   $$w_{\text{evidence}} = \min(0.85, 0.20 + 0.05 \cdot N_s)$$
   $$R_s^{(t)} = (1 - w_{\text{evidence}}) R_0 + w_{\text{evidence}} \cdot (S_{\text{consensus}} \cdot S_{\text{rate}} \cdot S_{\text{stuck}})$$
   Sources with $R_s < 0.40$ are flagged as bad sources and effectively silenced.

---

### 2. Time-Series Reconstruction (Nadaraya-Watson Kernel Smoothing)
Given irregular observations $(t_i, y_i)$ with reliability weights $w_i = R_{s_i}$:
$$x(t) = \frac{\sum_i K_h(t - t_i) \cdot w_i \cdot y_i}{\sum_i K_h(t - t_i) \cdot w_i + \epsilon}$$
where $K_h(u) = \exp\left(-\frac{u^2}{2 h^2}\right)$ is the Gaussian kernel with bandwidth $h = 0.6\text{h}$.

---

### 3. Causal Edge Discovery via Differenced TLCC
To prevent non-stationary trend confounding where all rising signals correlate spuriously, TLCC is computed on discrete innovations (first differences):
$$\Delta x_u(t) = x_u(t) - x_u(t - \Delta t)$$
$$\Delta x_v(t) = x_v(t) - x_v(t - \Delta t)$$

For candidate lag $\tau = k \cdot \Delta t \in [\tau_{\min}, \tau_{\max}]$:
$$r_{u \to v}(\tau) = \frac{\sum_t (\Delta x_u(t) - \overline{\Delta x}_u)(\Delta x_v(t + \tau) - \overline{\Delta x}_v)}{\sqrt{\sum_t (\Delta x_u(t) - \overline{\Delta x}_u)^2 \sum_t (\Delta x_v(t + \tau) - \overline{\Delta x}_v)^2} + \epsilon}$$

1. **Directionality / Asymmetry**:
   $$r_{u \to v}(\tau^*) > r_{v \to u}(\tau^*) \cdot 1.05$$
2. **Fisher $z$-Transformation Confidence**:
   $$z = \frac{1}{2}\ln\left(\frac{1 + r^*}{1 - r^*}\right), \quad \text{SE}_z = \frac{1}{\sqrt{N_{\text{eff}} - 3}}$$
   $$Z_{\text{stat}} = \frac{|z|}{\text{SE}_z}, \quad \text{Confidence} = 2 \cdot \Phi(Z_{\text{stat}}) - 1.0$$
3. **Transitive Reduction**:
   If $u \to w$ (lag $\tau_1$) and $w \to v$ (lag $\tau_2$) exist, and candidate edge $u \to v$ has $\tau_{uv} \approx \tau_1 + \tau_2$, the shortcut $u \to v$ is pruned to isolate the direct causal link.

---

### 4. Bayesian 4-State Probability Simplex
For zone $z$, aggregate stress is weighted across systems:
$$S_z(t) = \sum_{m=1}^{6} w_m x_{z, m}(t)$$
Effective stress incorporates dynamic forward momentum and incoming cascade load:
$$\tilde{S}_z = S_z + 0.25 \frac{dS_z}{dt} + 0.30 P_z^{\text{incoming}}$$

Logits are centered at intervals $\mu = [0.125, 0.375, 0.625, 0.875]$:
$$\text{logit}_k = -\frac{(\tilde{S}_z - \mu_k)^2}{2 \sigma^2} + \text{bias}_k\left(\frac{dS_z}{dt}\right)$$
$$P_k = \frac{\exp(\text{logit}_k - \max_j \text{logit}_j)}{\sum_{m=0}^3 \exp(\text{logit}_m - \max_j \text{logit}_j)}$$
Guarantees $\sum_{k=0}^3 P_k = 1.0$ and $P_k \ge 0$.

---

### 5. Time-to-Critical (TTC) Forward Simulation
Node state evolution is projected forward using coupled non-linear ODEs:
$$\frac{dx_v}{dt} = -\lambda_v x_v + \sum_{(u \to v) \in E} w_{uv} \frac{x_u(t - \tau_{uv})^2}{0.25 + x_u(t - \tau_{uv})^2}$$
Earliest horizon $h^*$ where projected aggregate stress $S_z(\text{now} + h^*) \ge 0.75$ is returned as `time_to_critical_h`. Inactive zones without elevated stress or incoming cascade pressure return `null`.

---

## Project Structure

```
Cascade72/
├── cascade72/
│   ├── __init__.py           # Package exports
│   ├── config.py             # System constants, zones, weights, thresholds
│   ├── math_utils.py         # Pure NumPy stats: TLCC, Fisher z, kernel smoothing, softmax
│   ├── source_filter.py      # Source reliability scoring, Byzantine filter, robust fusion
│   ├── graph_discovery.py    # Lagged cross-correlation, edge discovery, transitive reduction
│   ├── state_estimator.py    # Zone stress aggregation, 4-state probability simplex
│   ├── cascade_forecaster.py # Forward ODE network simulation, TTC forecasting, chain trace
│   ├── intervention_engine.py# Bottleneck leverage ranking & targeted mitigations
│   ├── engine.py             # Master Cascade72 step engine adhering to JSON output schema
│   ├── replay.py             # Chronological replay harness across [0..72h]
│   └── evaluator.py          # Ground truth benchmarking (F1, Precision, Recall, MAE)
├── data/
│   ├── cityA_events.csv      # Out-of-order delayed telemetry stream (City A)
│   ├── cityA_truth.json      # Ground truth causal graph, critical times, bad sources
│   ├── cityB_events.csv      # Out-of-order delayed telemetry stream (City B)
│   └── cityB_truth.json      # Ground truth benchmark (City B)
├── output/
│   └── engine_output.json    # Standardized engine output recomputed at each step
├── scripts/
│   ├── generate_datasets.py  # Generates realistic physics-based benchmark datasets
│   └── run_cascade72.py      # CLI runner for replay, step analysis, and dashboard
└── tests/                    # Comprehensive unit and integration test suite
    ├── test_math_utils.py
    ├── test_source_filter.py
    ├── test_graph_discovery.py
    ├── test_state_estimator.py
    ├── test_cascade_forecaster.py
    ├── test_intervention_engine.py
    ├── test_engine_integration.py
    └── test_replay_pipeline.py
```

---

## Quickstart & Usage

### 1. Run Engine at a Specific Timestamp
Evaluate City A at $t = 18.0$ hours:
```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe scripts\run_cascade72.py --events data/cityA_events.csv --now 18.0 --output output/engine_output.json --truth data/cityA_truth.json
```

### 2. Run Full 72-Hour Replay
Simulate chronological event stream playback from $t = 0.0$ to $t = 72.0$ with 6-hour increments:
```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe scripts\run_cascade72.py --events data/cityA_events.csv --replay --step 6.0 --truth data/cityA_truth.json
```

### 3. Run Unit & Integration Test Suite
```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m pytest tests -v
```

---

## Benchmark Results

### City A (Coastal Inundation Cascade) at T = 18.0h:
- **Composite Score**: `0.818`
- **Bad Sources Detection**: Precision: `1.000` | Recall: `1.000` | F1: `1.000` (detected `['src_04', 'src_09', 'src_15']`)
- **Causal Edge Discovery**: F1: `0.7826` | Recall: `0.900` | Lag MAE: `0.94 hours`
- **Critical Zones**: Z1 (`CRITICAL`), Z2 (`CRITICAL`) | Quiet Zones: Z3..Z6 (`NORMAL`, TTC: `null`)

### City B (Power Substation Collapse) at T = 24.0h:
- **Composite Score**: `0.9368`
- **Bad Sources Detection**: Precision: `1.000` | Recall: `1.000` | F1: `1.000` (detected `['src_02', 'src_07', 'src_11']`)
- **Causal Edge Discovery**: F1: `0.8421` | Recall: `1.000` (8 of 8 ground truth edges recovered) | Lag MAE: `1.12 hours`
- **Critical Zones**: Z3 (`CRITICAL`), Z4 (`CRITICAL`), Z5 (`HIGH_RISK`) | Quiet Zones: Z1, Z2, Z6 (`NORMAL`, TTC: `null`)
