"""Condition Analyzer — the core of the framework.

Every metric can be computed under different conditions (low/mid/high
uncertainty, ID vs OOD) without rewriting metric code.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List
from metrics.accumulator import StatsAccumulator


class ConditionAnalyzer:
    """Registry of named condition functions.

    Each condition is a callable ``stats → bool_mask``.
    """

    def __init__(self):
        self._conditions: Dict[str, Callable[[StatsAccumulator], np.ndarray]] = {
            "overall": lambda s: np.ones(len(s.error_A), dtype=bool),
        }
        self._thresholds: Dict[str, str] = {}  # human-readable threshold info

    def register(self, name: str, fn: Callable[[StatsAccumulator], np.ndarray]):
        """Add a named condition."""
        self._conditions[name] = fn

    def auto_register(self, stats: StatsAccumulator):
        """Register data-driven conditions based on available fields."""
        if stats.epi_var_A is not None and len(stats.epi_var_A) > 0:
            epi = stats.epi_var_A.flatten()
            p33, p66 = np.percentile(epi, [33, 66])
            self._thresholds["low_epi"]  = f"epi_std < {p33:.6f}"
            self._thresholds["mid_epi"]  = f"epi_std ∈ [{p33:.6f}, {p66:.6f})"
            self._thresholds["high_epi"] = f"epi_std >= {p66:.6f}"
            self.register("low_epi",  lambda s, lo=p33: (s.epi_var_A.flatten() < lo))
            self.register("mid_epi",  lambda s, lo=p33, hi=p66: ((s.epi_var_A.flatten() >= lo) & (s.epi_var_A.flatten() < hi)))
            self.register("high_epi", lambda s, hi=p66: (s.epi_var_A.flatten() >= hi))
        if stats.entropy_A is not None and len(stats.entropy_A) > 0:
            p33e, p66e = np.percentile(stats.entropy_A, [33, 66])
            self._thresholds["low_entropy"]  = f"entropy < {p33e:.6f}"
            self._thresholds["mid_entropy"]  = f"entropy ∈ [{p33e:.6f}, {p66e:.6f})"
            self._thresholds["high_entropy"] = f"entropy >= {p66e:.6f}"
            self.register("low_entropy",  lambda s, lo=p33e: (s.entropy_A < lo))
            self.register("mid_entropy",  lambda s, lo=p33e, hi=p66e: ((s.entropy_A >= lo) & (s.entropy_A < hi)))
            self.register("high_entropy", lambda s, hi=p66e: (s.entropy_A >= hi))

    def get_threshold_info(self) -> Dict[str, str]:
        """Return human-readable threshold descriptions for registered conditions."""
        return dict(self._thresholds)

    def registered(self) -> List[str]:
        """Return sorted list of registered condition names."""
        return sorted(self._conditions.keys())

    def get(self, name: str) -> Callable[[StatsAccumulator], np.ndarray] | None:
        """Get a condition function by name, or None."""
        return self._conditions.get(name)

    def compute_all(
        self,
        metric_fn: Callable[[StatsAccumulator], dict],
        stats: StatsAccumulator,
        enabled: List[str] | None = None,
    ) -> dict:
        """Run *metric_fn* on every enabled condition, return {cond_name: result}."""
        names = enabled or self.registered()
        results = {}
        for name in names:
            fn = self._conditions.get(name)
            if fn is not None:
                mask = fn(stats)
                if mask.sum() > 0:
                    results[name] = metric_fn(stats.sub_mask(mask))
                else:
                    results[name] = {}
        return results
