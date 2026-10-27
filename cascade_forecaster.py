"""Forward cascade forecasting, time-to-critical prediction, and causal chain extraction.

Simulates forward network propagation over discovered causal edges to project
when each zone will reach CRITICAL state (stress >= 0.75), and extracts the
dominant causal cascade chains driving the failure.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from cascade72.config import (
    ZONES,
    SYSTEMS,
    ALL_NODES,
    TOTAL_HORIZON_H,
    STRESS_THRESHOLDS,
    STATE_CRITICAL,
)
from cascade72.state_estimator import StateEstimator


class CascadeForecaster:
    """Projects future stress trajectories and traces causal cascade chains."""

    def __init__(self, state_estimator: Optional[StateEstimator] = None) -> None:
        self.estimator = state_estimator or StateEstimator()

    def forecast_time_to_critical(
        self,
        now: float,
        zone: str,
        node_timeseries: Dict[str, np.ndarray],
        discovered_edges: List[Dict[str, Any]],
        dt: float,
        zone_eval: Dict[str, Any],
    ) -> Optional[float]:
        """Calculates time_to_critical_h (hours from 'now' until zone reaches CRITICAL).

        Mathematical Formulation:
            1. If zone is currently CRITICAL (stress >= 0.75):
               time_to_critical_h = 0.0.

            2. Inactive Zone Filter:
               A zone only experiences impending failure if:
               - Current stress S_z >= 0.18, OR
               - Positive rate of change dS_z/dt > 0.015/h, OR
               - Active incoming cascade pressure from discovered edges P_incoming >= 0.15.
               Zones without elevated stress or incoming cascade pressure remain stable
               throughout the 72h window: returns None (rendered as null).

            3. Network Cascade ODE Simulation:
               For forward horizon h ∈ [dt, 72 - now]:
               Simulates node stress propagation:
                   dx_v/dt = -λ_v * x_v(t) + Σ_{(u->v)} w_{uv} * x_u(t - τ_{uv})
               where dissipation λ_v = 0.03/h, and w_{uv} = strength_{uv}.
               Finds earliest h* where S_z(now + h*) >= 0.75.

            4. First-order Momentum Extrapolation:
               If dS_z/dt > 0.015:
                   h_lin = (0.75 - S_z(now)) / (dS_z/dt).
               Combined projection: h* = min(h_sim, h_lin).

            5. If now + h* > 72.0:
               Returns None.
        """
        curr_stress = zone_eval["stress"]
        if curr_stress >= STRESS_THRESHOLDS[STATE_CRITICAL]:
            return 0.0

        remaining_h = TOTAL_HORIZON_H - now
        if remaining_h <= 0.0:
            return None

        rate = zone_eval.get("rate", 0.0)
        pressure = zone_eval.get("pressure", 0.0)

        # Check if zone is at risk of cascading failure
        if curr_stress < 0.18 and rate < 0.015 and pressure < 0.15:
            return None

        # Build simulation buffer from observed node history
        sim_history: Dict[str, List[float]] = {}
        for n in ALL_NODES:
            ts = node_timeseries.get(n)
            if ts is not None and len(ts) > 0:
                sim_history[n] = ts.tolist()
            else:
                sim_history[n] = [0.0]

        steps_forward = int(np.ceil(remaining_h / dt))
        dissipation = 0.02
        crit_threshold = STRESS_THRESHOLDS[STATE_CRITICAL]

        sim_critical_h: Optional[float] = None

        # Step forward in time
        for step in range(1, steps_forward + 1):
            h = step * dt
            step_updates: Dict[str, float] = {}

            for v in ALL_NODES:
                curr_v = sim_history[v][-1]
                incoming = 0.0

                for edge in discovered_edges:
                    if edge["to"] == v:
                        u = edge["from"]
                        lag = edge["lag_hours"]
                        strength = edge["strength"]

                        lag_steps_back = int(round(lag / dt))
                        idx = len(sim_history[u]) - lag_steps_back
                        if idx >= 0:
                            u_val = sim_history[u][idx]
                        else:
                            u_val = sim_history[u][0]

                        incoming += strength * (u_val ** 2) / (0.25 + u_val ** 2)

                delta = dt * (-dissipation * curr_v + incoming * 0.45)
                step_updates[v] = float(np.clip(curr_v + delta, 0.0, 1.0))

            for n, val in step_updates.items():
                sim_history[n].append(val)

            sim_node_ts = {n: np.array(sim_history[n]) for n in ALL_NODES}
            proj_stress = self.estimator.estimate_zone_stress(
                zone, sim_node_ts, time_idx=-1
            )

            if proj_stress >= crit_threshold:
                sim_critical_h = float(h)
                break

        # Linear momentum extrapolation
        lin_critical_h: Optional[float] = None
        if rate > 0.015:
            delta_needed = crit_threshold - curr_stress
            h_lin = delta_needed / rate
            if h_lin > 0.0 and (now + h_lin) <= TOTAL_HORIZON_H:
                lin_critical_h = float(h_lin)

        candidates = [h for h in [sim_critical_h, lin_critical_h] if h is not None]
        if candidates:
            best_h = min(candidates)
            if (now + best_h) <= TOTAL_HORIZON_H:
                return round(best_h, 2)

        return None

    def extract_cascade_chain(
        self,
        zone: str,
        node_timeseries: Dict[str, np.ndarray],
        discovered_edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extracts the dominant directed causal chain driving stress in this zone."""
        zone_nodes = {f"{zone}:{s}" for s in SYSTEMS}
        relevant_edges = [
            e for e in discovered_edges
            if e["to"] in zone_nodes or e["from"] in zone_nodes
        ]

        if not relevant_edges:
            return []

        active_chain: List[Dict[str, Any]] = []
        visited_edges = set()

        sorted_edges = sorted(
            relevant_edges,
            key=lambda e: e["strength"] * e["confidence"],
            reverse=True,
        )

        for edge in sorted_edges:
            u, v = edge["from"], edge["to"]
            edge_key = (u, v)
            if edge_key in visited_edges:
                continue

            u_ts = node_timeseries.get(u)
            v_ts = node_timeseries.get(v)
            u_val = float(u_ts[-1]) if u_ts is not None and len(u_ts) > 0 else 0.0
            v_val = float(v_ts[-1]) if v_ts is not None and len(v_ts) > 0 else 0.0

            if u_val > 0.15 or v_val > 0.15:
                visited_edges.add(edge_key)
                active_chain.append({
                    "from": u,
                    "to": v,
                    "lag_h": edge["lag_hours"],
                    "confidence": edge["confidence"],
                    "evidence": (
                        f"Cross-lagged transfer r={edge['strength']:.2f} at {edge['lag_hours']}h lag; "
                        f"source stress={u_val:.2f}, target stress={v_val:.2f}"
                    ),
                })
                if len(active_chain) >= 5:
                    break

        return active_chain
