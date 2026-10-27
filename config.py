"""Configuration and constants for Cascade72 urban failure forecasting engine.

Defines the 6 urban zones, 6 critical systems, 4-tier categorical states,
stress thresholds, and numerical hyperparameters for time-series discretization
and cross-correlation estimation.
"""

from typing import List, Dict, Tuple

# Urban spatial zones
ZONES: List[str] = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]

# Critical interconnected urban infrastructure systems
SYSTEMS: List[str] = [
    "rain",
    "drainage",
    "traffic",
    "ambulance",
    "hospital",
    "power",
]

# Total system nodes across all zones (36 nodes)
ALL_NODES: List[str] = [f"{z}:{s}" for z in ZONES for s in SYSTEMS]

# Discrete operational risk states
STATE_NORMAL: str = "NORMAL"
STATE_WATCH: str = "WATCH"
STATE_HIGH_RISK: str = "HIGH_RISK"
STATE_CRITICAL: str = "CRITICAL"

STATES: List[str] = [
    STATE_NORMAL,
    STATE_WATCH,
    STATE_HIGH_RISK,
    STATE_CRITICAL,
]

# State mapping indices for probability vectors [P(NORMAL), P(WATCH), P(HIGH_RISK), P(CRITICAL)]
STATE_TO_IDX: Dict[str, int] = {
    STATE_NORMAL: 0,
    STATE_WATCH: 1,
    STATE_HIGH_RISK: 2,
    STATE_CRITICAL: 3,
}
IDX_TO_STATE: Dict[int, str] = {v: k for k, v in STATE_TO_IDX.items()}

# Stress thresholds defining risk boundaries: [0.0, 1.0]
# Normal: [0.0, 0.25), Watch: [0.25, 0.50), High Risk: [0.50, 0.75), Critical: [0.75, 1.0]
STRESS_THRESHOLDS: Dict[str, float] = {
    STATE_NORMAL: 0.0,
    STATE_WATCH: 0.25,
    STATE_HIGH_RISK: 0.50,
    STATE_CRITICAL: 0.75,
}

# System importance weights for zone risk aggregation
# Hospital, power, drainage have higher weight in critical cascading scenarios
SYSTEM_WEIGHTS: Dict[str, float] = {
    "rain": 0.10,
    "drainage": 0.20,
    "traffic": 0.15,
    "ambulance": 0.15,
    "hospital": 0.20,
    "power": 0.20,
}

# Temporal resolution and horizons (in hours)
TOTAL_HORIZON_H: float = 72.0
TIME_STEP_H: float = 0.5       # Discretization grid step for time series
MIN_LAG_H: float = 0.5         # Minimum causal delay allowed between nodes
MAX_LAG_H: float = 12.0        # Maximum causal cascade window evaluated
KERNEL_BANDWIDTH_H: float = 1.0 # Gaussian kernel smoothing bandwidth

# Causal Discovery thresholds
MIN_CORRELATION_THRESHOLD: float = 0.35  # Minimum Pearson r to declare candidate edge
MIN_CONFIDENCE_THRESHOLD: float = 0.40   # Minimum statistical confidence [0, 1]
MIN_DATA_POINTS: int = 6                # Minimum observations required for regression/TLCC

# Source Reliability parameters
INITIAL_SOURCE_RELIABILITY: float = 0.80
BAD_SOURCE_THRESHOLD: float = 0.40      # Below this threshold, source is flagged bad
CONSENSUS_SIGMA: float = 0.15           # Residual tolerance for peer agreement
STUCK_SENSOR_VAR_THRESHOLD: float = 1e-4 # Minimum variance to avoid stuck sensor penalty
MAX_PHYSICAL_RATE: float = 0.70         # Max plausible delta stress per hour
