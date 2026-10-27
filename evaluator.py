"""Ground truth scoring and benchmarking module for Cascade72.

Used EXCLUSIVELY for evaluation and scoring. Never accessed by the engine.
Evaluates:
  1. Causal edge discovery Precision, Recall, F1, and Lag MAE.
  2. Bad source classification accuracy, Precision, and Recall.
  3. Time-to-critical prediction Mean Absolute Error (MAE) and early warning lead time.
"""

from typing import Dict, List, Optional, Any
import json
import numpy as np


class GroundTruthEvaluator:
    """Computes benchmark metrics comparing engine outputs against ground truth."""

    def __init__(self, truth_path: str) -> None:
        with open(truth_path, "r", encoding="utf-8") as f:
            self.truth: Dict[str, Any] = json.load(f)

        self.true_edges: List[Dict[str, Any]] = self.truth.get("edges", [])
        self.true_critical_times: Dict[str, Optional[float]] = self.truth.get("critical_time", {})
        self.true_bad_sources: List[str] = [str(s) for s in self.truth.get("bad_sources", [])]

    def score_bad_sources(
        self,
        predicted_reliabilities: Dict[str, float],
        threshold: float = 0.40,
    ) -> Dict[str, float]:
        """Evaluates classification accuracy of malicious/broken/noisy sources."""
        pred_bad = {s for s, r in predicted_reliabilities.items() if r < threshold}
        true_bad = set(self.true_bad_sources)

        tp = len(pred_bad & true_bad)
        fp = len(pred_bad - true_bad)
        fn = len(true_bad - pred_bad)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not true_bad else 0.0)
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "bad_sources_precision": round(precision, 4),
            "bad_sources_recall": round(recall, 4),
            "bad_sources_f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    @staticmethod
    def _edges_match(t_from: str, t_to: str, p_from: str, p_to: str) -> bool:
        """Determines if a predicted edge matches a ground truth edge.

        Handles both exact node matching (e.g. 'Z1:rain' == 'Z1:rain')
        and system-level matching (e.g. 'rain' == 'Z1:rain').
        """
        if t_from == p_from and t_to == p_to:
            return True

        t_u_sys = t_from.split(":")[-1]
        t_v_sys = t_to.split(":")[-1]
        p_u_sys = p_from.split(":")[-1]
        p_v_sys = p_to.split(":")[-1]

        if t_u_sys == p_u_sys and t_v_sys == p_v_sys:
            # If both specify a zone prefix, ensure zones match
            if ":" in t_from and ":" in p_from:
                return t_from.split(":")[0] == p_from.split(":")[0]
            return True

        return False

    def score_discovered_edges(
        self,
        discovered_edges: List[Dict[str, Any]],
        lag_tolerance_h: float = 2.0,
    ) -> Dict[str, float]:
        """Evaluates causal edge discovery against true causal graph."""
        if not self.true_edges:
            return {
                "edge_precision": 1.0 if not discovered_edges else 0.0,
                "edge_recall": 1.0,
                "edge_f1": 1.0,
                "lag_mae_hours": 0.0,
            }

        matched_true = set()
        matched_pred = set()
        lag_errors = []

        # Find best matches between true edges and predicted edges
        for t_idx, t_edge in enumerate(self.true_edges):
            t_u = t_edge["from"]
            t_v = t_edge["to"]
            t_lag = t_edge["lag_hours"]

            best_p_idx = None
            min_lag_err = float("inf")

            for p_idx, p_edge in enumerate(discovered_edges):
                if p_idx in matched_pred:
                    continue

                if self._edges_match(t_u, t_v, p_edge["from"], p_edge["to"]):
                    err = abs(p_edge["lag_hours"] - t_lag)
                    if err <= lag_tolerance_h and err < min_lag_err:
                        min_lag_err = err
                        best_p_idx = p_idx

            if best_p_idx is not None:
                matched_true.add(t_idx)
                matched_pred.add(best_p_idx)
                lag_errors.append(min_lag_err)

        tp = len(matched_true)
        fp = len(discovered_edges) - len(matched_pred)
        fn = len(self.true_edges) - len(matched_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        lag_mae = float(np.mean(lag_errors)) if lag_errors else 0.0

        return {
            "edge_precision": round(precision, 4),
            "edge_recall": round(recall, 4),
            "edge_f1": round(f1, 4),
            "lag_mae_hours": round(lag_mae, 4),
            "true_edges_count": len(self.true_edges),
            "discovered_edges_count": len(discovered_edges),
            "matched_edges_count": tp,
        }

    def score_time_to_critical(
        self,
        now: float,
        zones_output: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluates prediction error of time_to_critical_h at timestamp 'now'."""
        errors = []
        lead_times = []
        false_alarms = 0

        for z_out in zones_output:
            zone = z_out["zone"]
            pred_ttc = z_out.get("time_to_critical_h")
            true_crit_time = self.true_critical_times.get(zone)

            if true_crit_time is not None:
                true_remaining = max(0.0, true_crit_time - now)

                # Zone has not failed yet: evaluate prediction accuracy
                if now < true_crit_time:
                    if pred_ttc is not None:
                        err = abs(pred_ttc - true_remaining)
                        errors.append(err)
                        lead_times.append(pred_ttc)
            else:
                # Zone never fails in 72h
                if pred_ttc is not None and (now + pred_ttc) < 72.0:
                    false_alarms += 1

        mae = float(np.mean(errors)) if errors else 0.0
        mean_lead = float(np.mean(lead_times)) if lead_times else 0.0

        return {
            "time_to_critical_mae_h": round(mae, 4),
            "mean_early_warning_lead_h": round(mean_lead, 4),
            "evaluated_zones": len(errors),
            "false_alarms_count": false_alarms,
        }

    def evaluate_engine_output(self, engine_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generates comprehensive scoring report for a single engine output step."""
        now = engine_output["now"]
        bad_score = self.score_bad_sources(engine_output.get("source_reliability", {}))
        edge_score = self.score_discovered_edges(engine_output.get("discovered_edges", []))
        ttc_score = self.score_time_to_critical(now, engine_output.get("zones", []))

        # Composite benchmark score
        composite = float(
            0.40 * edge_score["edge_f1"]
            + 0.35 * bad_score["bad_sources_f1"]
            + 0.25 * max(0.0, 1.0 - ttc_score["time_to_critical_mae_h"] / 10.0)
        )

        return {
            "now": now,
            "composite_score": round(composite, 4),
            "edge_metrics": edge_score,
            "source_metrics": bad_score,
            "timing_metrics": ttc_score,
        }
