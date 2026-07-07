"""Direction consistency of 3D error vectors between two models."""

from __future__ import annotations
import numpy as np


def compute(stats) -> dict:
    """Compute direction-consistency metrics.

    Args:
        stats: ``StatsAccumulator`` with ``diff_A`` and ``diff_B`` as
            numpy arrays of shape ``(N, 3)``.

    Returns:
        dict with ``cosine_similarity_mean``, ``cosine_similarity_std``,
        ``sign_agreement_x/y/z``, ``n_pixels``.
    """
    dA = stats.diff_A
    dB = stats.diff_B
    N = len(dA)
    if N == 0:
        return {"cosine_similarity_mean": None, "n_pixels": 0}

    # per-axis sign agreement
    signs_A = np.sign(dA)
    signs_B = np.sign(dB)
    agree = (signs_A == signs_B).mean(axis=0)

    # cosine similarity
    norm_A = np.linalg.norm(dA, axis=1) + 1e-12
    norm_B = np.linalg.norm(dB, axis=1) + 1e-12
    cos_sim = (dA * dB).sum(axis=1) / (norm_A * norm_B)

    return {
        "cosine_similarity_mean": float(cos_sim.mean()),
        "cosine_similarity_std":  float(cos_sim.std()),
        "sign_agreement_x": float(agree[0]),
        "sign_agreement_y": float(agree[1]),
        "sign_agreement_z": float(agree[2]),
        "n_pixels": int(N),
    }
