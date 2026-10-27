"""Causal cascade graph discovery via directional Time-Lagged Cross-Correlation (TLCC).

Identifies directed dependencies between urban infrastructure systems and zones:
e.g. 'rain' -> 'drainage' (lag 3h), 'drainage' -> 'traffic' (lag 2h).
Uses pure NumPy statistical methods with differenced impulse response and Fisher z-confidence tests.
"""

from typing import Dict, List, Tuple, Any, Set
import numpy as np

from cascade72.config import (
    ALL_NODES,
    ZONES,
    SYSTEMS,
    MIN_LAG_H,
    MAX_LAG_H,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_DATA_POINTS,
)
from cascade72.math_utils import (
    time_lagged_cross_correlation,
    estimate_edge_strength_and_confidence,
)


class CausalGraphDiscoverer:
    """Discovers directed causal edges and transmission lags between urban system nodes.

    Mathematical Principles:
        1. Innovation Differencing (Stationarity & Impulse Response):
           Cumulative non-stationary stress profiles create spurious trend correlations.
           We correlate the discrete first differences (innovations):
               Δx_u(t) = x_u(t) - x_u(t - Δt)
               Δx_v(t) = x_v(t) - x_v(t - Δt)
           An impulse in Δx_u at t that triggers an impulse in Δx_v at t + τ produces
           a sharp, unambiguous cross-correlation peak at physical lag τ* > 0.

        2. Physical Urban Network Topology:
           Validates candidate pairs:
           - Intra-zone cascades: any distinct system pairs within the same zone.
           - Inter-zone cascades: spatial grid or infrastructure flow between zones.

        3. Asymmetry & Directionality Verification:
           Physical propagation is asymmetric:
               r_{u→v}(τ*) > r_{v→u}(τ*)
           If forward correlation is not significantly greater than reverse correlation,
           the co-movement is discarded as non-causal confounding.

        4. Transitive Reduction:
           If u -> w (lag τ_1) and w -> v (lag τ_2) exist, a candidate edge u -> v with
           lag ≈ τ_1 + τ_2 is recognized as an indirect multi-hop path and pruned.
    """

    def __init__(
        self,
        min_lag_h: float = MIN_LAG_H,
        max_lag_h: float = 8.0,
        min_corr: float = 0.55,
        min_conf: float = MIN_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.min_lag_h = min_lag_h
        self.max_lag_h = max_lag_h
        self.min_corr = min_corr
        self.min_conf = min_conf

    def is_physical_candidate(self, u: str, v: str) -> bool:
        """Enforces urban domain topology on candidate causal pairs."""
        if u == v:
            return False
        u_zone, u_sys = u.split(":")
        v_zone, v_sys = v.split(":")

        # Intra-zone physical coupling: any two distinct systems in the same zone
        if u_zone == v_zone:
            return u_sys != v_sys

        # Inter-zone physical coupling: same system across zones (drainage runoff, power grid, etc.)
        if u_sys == v_sys:
            return True

        return False

    def discover_edges(
        self,
        node_timeseries: Dict[str, np.ndarray],
        dt: float,
    ) -> List[Dict[str, Any]]:
        """Analyzes all candidate node pairs to discover active causal cascade edges.

        Args:
            node_timeseries: Mapping node_name -> 1D smoothed stress array.
            dt: Time step size of the regularized grid (hours).

        Returns:
            List of discovered edges formatted as:
            [
                {
                    "from": "Z1:drainage",
                    "to": "Z1:traffic",
                    "lag_hours": 2.0,
                    "strength": 0.85,
                    "confidence": 0.90
                },
                ...
            ]
        """
        discovered: List[Dict[str, Any]] = []

        # Identify active nodes with non-trivial stress excursions
        active_nodes = {
            n: ts for n, ts in node_timeseries.items()
            if len(ts) >= MIN_DATA_POINTS and np.max(ts) >= 0.15 and np.std(np.diff(ts)) >= 0.006
        }

        if len(active_nodes) < 2:
            return discovered

        raw_candidates: List[Dict[str, Any]] = []

        for u, x in active_nodes.items():
            dx = np.diff(x)
            for v, y in active_nodes.items():
                if not self.is_physical_candidate(u, v):
                    continue

                dy = np.diff(y)

                # Time-lagged cross-correlation on first differences
                lag_h, r_forward, eff_n = time_lagged_cross_correlation(
                    dx, dy, dt=dt, min_lag_h=self.min_lag_h, max_lag_h=self.max_lag_h
                )

                if r_forward < self.min_corr or eff_n < MIN_DATA_POINTS:
                    continue

                # Enforce directional causality: forward correlation must exceed reverse
                _, r_reverse, _ = time_lagged_cross_correlation(
                    dy, dx, dt=dt, min_lag_h=self.min_lag_h, max_lag_h=self.max_lag_h
                )

                if r_forward < r_reverse * 1.05:
                    continue

                lag_steps = int(round(lag_h / dt))
                strength, confidence = estimate_edge_strength_and_confidence(
                    x, y, lag_steps=lag_steps, correlation=r_forward, effective_n=eff_n
                )

                if confidence >= self.min_conf and strength >= 0.20:
                    raw_candidates.append({
                        "from": u,
                        "to": v,
                        "lag_hours": round(float(lag_h), 2),
                        "strength": round(float(strength), 4),
                        "confidence": round(float(confidence), 4),
                        "correlation": round(float(r_forward), 4),
                    })

        # Transitive Reduction: Prune indirect 2-hop paths
        edge_map = {(e["from"], e["to"]): e for e in raw_candidates}
        pruned_edges: List[Dict[str, Any]] = []

        for edge in raw_candidates:
            u, v = edge["from"], edge["to"]
            uv_lag = edge["lag_hours"]
            is_transitive = False

            for (e1_from, e1_to), e1 in edge_map.items():
                if e1_from == u and e1_to != v:
                    w = e1_to
                    if (w, v) in edge_map:
                        e2 = edge_map[(w, v)]
                        combined_lag = e1["lag_hours"] + e2["lag_hours"]
                        if abs(uv_lag - combined_lag) <= (dt * 2.0):
                            is_transitive = True
                            break

            if not is_transitive:
                pruned_edges.append(edge)

        # Sort by strength * confidence descending
        pruned_edges.sort(
            key=lambda e: e["strength"] * e["confidence"],
            reverse=True,
        )

        for e in pruned_edges:
            discovered.append({
                "from": e["from"],
                "to": e["to"],
                "lag_hours": e["lag_hours"],
                "strength": e["strength"],
                "confidence": e["confidence"],
            })

        return discovered
