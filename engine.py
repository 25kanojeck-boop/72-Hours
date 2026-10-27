"""Cascade72 Core Forecasting Engine.

Reconstructs delayed out-of-order event streams, downweights corrupted sources,
discovers physical causal cascade networks, estimates zone state probabilities,
projects time-to-critical horizons, and outputs comprehensive risk intelligence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import pandas as pd

from cascade72.config import (
    ZONES,
    SYSTEMS,
    ALL_NODES,
    TIME_STEP_H,
    KERNEL_BANDWIDTH_H,
    TOTAL_HORIZON_H,
)
from cascade72.math_utils import kernel_smooth_timeseries
from cascade72.source_filter import SourceReliabilityFilter
from cascade72.graph_discovery import CausalGraphDiscoverer
from cascade72.state_estimator import StateEstimator
from cascade72.cascade_forecaster import CascadeForecaster
from cascade72.intervention_engine import InterventionEngine


class CascadeEngine:
    """Master engine executing the 72-hour cascading failure prediction pipeline."""

    def __init__(
        self,
        dt: float = TIME_STEP_H,
        bandwidth: float = KERNEL_BANDWIDTH_H,
        output_path: str = "output/engine_output.json",
    ) -> None:
        self.dt = dt
        self.bandwidth = bandwidth
        self.output_path = Path(output_path)

        # Submodules
        self.source_filter = SourceReliabilityFilter()
        self.graph_discoverer = CausalGraphDiscoverer()
        self.state_estimator = StateEstimator()
        self.cascade_forecaster = CascadeForecaster(self.state_estimator)
        self.intervention_engine = InterventionEngine()

    def step(
        self,
        now: float,
        full_events_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Executes a single replay step at timestamp 'now'.

        Strict Causality Guarantee:
            Only events where arrival_time <= now are ingested. Events with
            arrival_time > now are in the future and strictly invisible.

        Args:
            now: Current simulation time (hours, 0.0 to 72.0).
            full_events_df: Full event stream DataFrame.

        Returns:
            Dictionary matching the exact ENGINE OUTPUT schema.
        """
        # 1. Temporal filter: Arrived events only
        arrived_mask = full_events_df["arrival_time"] <= (now + 1e-6)
        visible_events = full_events_df[arrived_mask].copy()

        # 2. Dynamic Source Reliability Update
        source_reliabilities = self.source_filter.update_with_events(visible_events)

        # 3. Time-Series Reconstruction on regular grid [0, now]
        # Generate grid with step self.dt
        grid_points = max(2, int(np.floor(now / self.dt)) + 1)
        target_grid = np.linspace(0.0, max(now, self.dt), grid_points)

        node_timeseries: Dict[str, np.ndarray] = {}

        if not visible_events.empty:
            # Map source weights
            visible_events["source_weight"] = visible_events["source_id"].astype(str).map(
                lambda s: self.source_filter.get_reliability(s)
            )

            # Reconstruct each node
            for zone in ZONES:
                for system in SYSTEMS:
                    node_key = f"{zone}:{system}"
                    mask = (visible_events["zone"] == zone) & (visible_events["system"] == system)
                    sub = visible_events[mask]

                    if not sub.empty:
                        times = sub["event_time"].to_numpy(dtype=np.float64)
                        vals = sub["value"].to_numpy(dtype=np.float64)
                        weights = sub["source_weight"].to_numpy(dtype=np.float64)

                        smoothed = kernel_smooth_timeseries(
                            sample_times=times,
                            sample_values=vals,
                            sample_weights=weights,
                            target_grid=target_grid,
                            bandwidth=self.bandwidth,
                        )
                        node_timeseries[node_key] = smoothed
                    else:
                        node_timeseries[node_key] = np.zeros_like(target_grid)
        else:
            for n in ALL_NODES:
                node_timeseries[n] = np.zeros_like(target_grid)

        # 4. Causal Cascade Graph Discovery
        discovered_edges = self.graph_discoverer.discover_edges(
            node_timeseries=node_timeseries,
            dt=self.dt,
        )

        # 5. Zone State & Probabilities Estimation
        city_state, zone_evals = self.state_estimator.evaluate_all_zones(
            node_timeseries=node_timeseries,
            discovered_edges=discovered_edges,
            dt=self.dt,
        )

        # 6. Forecasting & Chain Extraction per Zone
        zones_output: List[Dict[str, Any]] = []
        for zone in ZONES:
            z_eval = zone_evals[zone]

            # Forecast time to critical
            ttc = self.cascade_forecaster.forecast_time_to_critical(
                now=now,
                zone=zone,
                node_timeseries=node_timeseries,
                discovered_edges=discovered_edges,
                dt=self.dt,
                zone_eval=z_eval,
            )

            # Extract causal chain
            chain = self.cascade_forecaster.extract_cascade_chain(
                zone=zone,
                node_timeseries=node_timeseries,
                discovered_edges=discovered_edges,
            )

            zones_output.append({
                "zone": zone,
                "state": z_eval["state"],
                "probs": z_eval["probs"],
                "time_to_critical_h": ttc,
                "chain": chain,
            })

        # 7. Actionable Interventions
        interventions = self.intervention_engine.generate_interventions(
            zones_output=zones_output,
            discovered_edges=discovered_edges,
            node_timeseries=node_timeseries,
        )

        # 8. Assemble ENGINE OUTPUT payload
        engine_output: Dict[str, Any] = {
            "now": round(float(now), 2),
            "city_state": city_state,
            "zones": zones_output,
            "discovered_edges": discovered_edges,
            "source_reliability": source_reliabilities,
            "interventions": interventions,
        }

        # 9. Persist to disk
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(engine_output, f, indent=2)

        return engine_output
