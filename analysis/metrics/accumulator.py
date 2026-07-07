"""StatsAccumulator — unified per-pixel statistics container for analysis."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StatsAccumulator:
    """Collects per-pixel predictions from two models for analysis.

    All fields are Python lists during collection (one entry per image),
    then converted to concatenated numpy arrays via ``finalize()``.

    Use ``sub_mask(condition_mask)`` to obtain a filtered copy for
    condition-conditioned metric computation.
    """

    # ---- per-pixel scalar errors ----
    error_A: list = field(default_factory=list)   # L2 error of model A
    error_B: list = field(default_factory=list)

    # ---- per-pixel 3D error vectors ----
    diff_A: list = field(default_factory=list)    # coords_A - GT  (N, 3)
    diff_B: list = field(default_factory=list)

    # ---- per-pixel 3D predictions ----
    coord_A: list = field(default_factory=list)   # predicted coords  (N, 3)
    coord_B: list = field(default_factory=list)
    coord_gt: list = field(default_factory=list)  # ground truth      (N, 3)

    # ---- NIG uncertainty (DER only, None for other types) ----
    epi_var_A: list = field(default_factory=list)   # epistemic variance
    alea_var_A: list = field(default_factory=list)  # aleatoric variance

    # ---- Bin distribution (gs only, None for other types) ----
    bin_probs_A: list = field(default_factory=list)  # per-pixel softmax  (N, K)
    bin_centers: Optional[np.ndarray] = None          # shared bin centers   (K,)
    entropy_A: list = field(default_factory=list)    # per-pixel entropy

    # ---- metadata ----
    split_name: str = ""

    def push(
        self,
        *,
        error_A: Optional[np.ndarray] = None,
        error_B: Optional[np.ndarray] = None,
        diff_A: Optional[np.ndarray] = None,
        diff_B: Optional[np.ndarray] = None,
        coord_A: Optional[np.ndarray] = None,
        coord_B: Optional[np.ndarray] = None,
        coord_gt: Optional[np.ndarray] = None,
        epi_var_A: Optional[np.ndarray] = None,
        alea_var_A: Optional[np.ndarray] = None,
        bin_probs_A: Optional[np.ndarray] = None,
        bin_centers: Optional[np.ndarray] = None,
        entropy_A: Optional[np.ndarray] = None,
    ):
        """Append per-image numpy arrays.  Pass ``None`` for unavailable fields."""
        if error_A is not None:
            self.error_A.append(error_A)
        if error_B is not None:
            self.error_B.append(error_B)
        if diff_A is not None:
            self.diff_A.append(diff_A)
        if diff_B is not None:
            self.diff_B.append(diff_B)
        if coord_A is not None:
            self.coord_A.append(coord_A)
        if coord_B is not None:
            self.coord_B.append(coord_B)
        if coord_gt is not None:
            self.coord_gt.append(coord_gt)
        if epi_var_A is not None:
            self.epi_var_A.append(epi_var_A)
        if alea_var_A is not None:
            self.alea_var_A.append(alea_var_A)
        if bin_probs_A is not None:
            self.bin_probs_A.append(bin_probs_A)
        if bin_centers is not None:
            self.bin_centers = bin_centers
        if entropy_A is not None:
            self.entropy_A.append(entropy_A)

    def finalize(self):
        """Convert list-of-arrays to single concatenated numpy arrays."""
        self.error_A = np.concatenate(self.error_A) if self.error_A else np.array([])
        self.error_B = np.concatenate(self.error_B) if self.error_B else np.array([])
        self.diff_A = np.concatenate(self.diff_A) if self.diff_A else np.empty((0, 3))
        self.diff_B = np.concatenate(self.diff_B) if self.diff_B else np.empty((0, 3))
        self.coord_A = np.concatenate(self.coord_A) if self.coord_A else np.empty((0, 3))
        self.coord_B = np.concatenate(self.coord_B) if self.coord_B else np.empty((0, 3))
        self.coord_gt = np.concatenate(self.coord_gt) if self.coord_gt else np.empty((0, 3))
        self.epi_var_A = np.concatenate(self.epi_var_A) if self.epi_var_A else None
        self.alea_var_A = np.concatenate(self.alea_var_A) if self.alea_var_A else None
        self.bin_probs_A = np.concatenate(self.bin_probs_A) if self.bin_probs_A else None
        self.entropy_A = np.concatenate(self.entropy_A) if self.entropy_A else None

    def sub_mask(self, mask: np.ndarray) -> "StatsAccumulator":
        """Return a new accumulator with only rows where *mask* is True."""
        if len(mask) == 0:
            return StatsAccumulator()
        def _maybe(arr, key):
            if arr is None:
                return None
            try:
                n_arr = len(arr)
            except TypeError:
                return arr
            if n_arr == 0:
                return arr
            try:
                return arr[mask]
            except IndexError:
                return np.array([])
        return StatsAccumulator(
            error_A=_maybe(self.error_A, 'error_A') if len(self.error_A) > 0 else np.array([]),
            error_B=_maybe(self.error_B, 'error_B') if len(self.error_B) > 0 else np.array([]),
            diff_A=_maybe(self.diff_A, 'diff_A'),
            diff_B=_maybe(self.diff_B, 'diff_B'),
            coord_A=_maybe(self.coord_A, 'coord_A'),
            coord_B=_maybe(self.coord_B, 'coord_B'),
            coord_gt=_maybe(self.coord_gt, 'coord_gt'),
            epi_var_A=_maybe(self.epi_var_A, 'epi_var_A'),
            alea_var_A=_maybe(self.alea_var_A, 'alea_var_A'),
            bin_probs_A=_maybe(self.bin_probs_A, 'bin_probs_A'),
            bin_centers=self.bin_centers,
            entropy_A=_maybe(self.entropy_A, 'entropy_A'),
            split_name=self.split_name,
        )
