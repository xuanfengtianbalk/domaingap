"""Pairwise coordinate comparison between two models.

Collects per-pixel predictions into a ``StatsAccumulator``, then delegates
metric computation to ``metrics/*.py`` modules.

Does NOT contain metric logic itself.
"""

from __future__ import annotations
import json, os, sys, numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import analysis_utils
from metrics.accumulator import StatsAccumulator


def _compute_errors(coords_a, coords_b, coords_gt, mask):
    """Compute per-pixel L2 errors and 3D error vectors for one image."""
    ca = coords_a[:, mask].numpy()
    cb = coords_b[:, mask].numpy()
    gt = coords_gt[:, mask].numpy()
    valid = np.isfinite(gt).all(axis=0)
    if not valid.any():
        return None
    ca, cb, gt = ca[:, valid], cb[:, valid], gt[:, valid]
    ea = np.linalg.norm(ca - gt, axis=0)
    eb = np.linalg.norm(cb - gt, axis=0)
    da = (ca - gt).T
    db = (cb - gt).T
    ca_t = ca.T
    cb_t = cb.T
    gt_t = gt.T
    return ea, eb, da, db, ca_t, cb_t, gt_t


def _extract_uncertainty(full: dict, mask, valid: np.ndarray) -> dict:
    """Extract per-pixel uncertainty values from model output dict."""
    extra = {}
    key_map = {'epi_var': 'epi_var_A', 'alea_var': 'alea_var_A'}
    for src_key, dst_key in key_map.items():
        val = full.get(src_key)
        if val is not None:
            arr = val[0] if val.ndim == 3 else val.squeeze(0)
            per_coord = arr[:, mask][:, valid]                                # [3, N']
            extra[dst_key] = np.sqrt((per_coord ** 2).sum(axis=0))            # [N'] sqrt(x²+y²+z²)
    return extra


def _extract_bins(full: dict, mask, valid: np.ndarray) -> dict:
    """Extract per-pixel bin distribution values."""
    extra = {}
    bp = full.get('bin_probs')
    if bp is not None:
        K = bp.shape[-1]
        arr = bp.squeeze(0).reshape(-1, 3, K)
        pix = arr[mask][valid]
        extra['bin_probs_A'] = pix.reshape(-1, K)
    ent = full.get('entropy')
    if ent is not None:
        arr = ent.squeeze(0)
        extra['entropy_A'] = arr[mask][valid].flatten()
    bc = full.get('bin_centers')
    if bc is not None:
        extra['bin_centers'] = bc
    return extra


def run_pairwise(uuid_1, uuid_2, splits, max_samples=None, batch_size=1, device='cuda:0',
                 metrics_enabled=None, conditions_enabled=None, visualization_enabled=None):
    """Main analysis entry point for pairwise comparison."""
    mt1, bb1 = analysis_utils.get_model_type_from_traininfo(uuid_1)
    mt2, bb2 = analysis_utils.get_model_type_from_traininfo(uuid_2)
    model_type_a = mt1[0]
    model_type_b = mt2[0]
    print(f"Model A ({uuid_1[:8]}): {model_type_a} on {bb1}")
    print(f"Model B ({uuid_2[:8]}): {model_type_b} on {bb2}")

    model_a, bc_a = analysis_utils.load_model(uuid_1, model_type_a, bb1, device)
    model_b, bc_b = analysis_utils.load_model(uuid_2, model_type_b, bb2, device)

    base_out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs',
                                            f'pairwise_{uuid_1[:8]}_{uuid_2[:8]}'))
    os.makedirs(base_out, exist_ok=True)

    # import metrics lazily, based on what is requested
    enabled = set(metrics_enabled or ['correlation', 'winrate', 'direction'])
    metric_modules = {}
    for m in enabled:
        try:
            metric_modules[m] = __import__(f'metrics.{m}', fromlist=['compute'])
        except (ImportError, ModuleNotFoundError):
            print(f"  [WARN] metric '{m}' not found, skipping")

    # condition analyzer (auto-register from data)
    from metrics.condition import ConditionAnalyzer
    cond_analyzer = ConditionAnalyzer()

    for split in splits:
        print(f"\n=== {split} ===")
        dl = analysis_utils.build_dataloader(uuid_1, split, max_samples=max_samples, batch_size=batch_size)

        # probe for mask_gt
        _, probe_targets = next(iter(dl))
        if 'mask_gt' not in probe_targets:
            print(f"  {split}: no mask_gt, skipping")
            continue

        dl = analysis_utils.build_dataloader(uuid_1, split, max_samples=max_samples, batch_size=batch_size)
        stats = StatsAccumulator(split_name=split)

        for samples, targets in dl:
            full_a = analysis_utils.extract_full(model_a, bc_a, samples, device, model_type_a)
            full_b = analysis_utils.extract_full(model_b, bc_b, samples, device, model_type_b)

            for b in range(samples.shape[0]):
                mask_gt = targets['mask_gt'][b] > 0.5
                if mask_gt.sum() == 0:
                    continue
                coords_gt = targets['coors_gt'][b]

                errs = _compute_errors(full_a['coords'][b], full_b['coords'][b], coords_gt, mask_gt)
                if errs is None:
                    continue
                ea, eb, da, db, ca, cb, gt = errs

                # basic
                stats.push(error_A=ea, error_B=eb, diff_A=da, diff_B=db,
                           coord_A=ca, coord_B=cb, coord_gt=gt)

                # uncertainty from model A
                idx = mask_gt.numpy()
                valid = np.isfinite(coords_gt.numpy()).all(axis=0)[idx]
                ua = _extract_uncertainty(full_a, idx, valid)
                stats.push(**ua)

                # bins from model A
                ba = _extract_bins(full_a, idx, valid)
                stats.push(**ba)

        stats.finalize()
        cond_analyzer.auto_register(stats)

        if not enabled:
            continue

        save_dir = os.path.join(base_out, split)
        os.makedirs(save_dir, exist_ok=True)

        for metric_name in sorted(enabled):
            if metric_name not in metric_modules:
                continue
            compute_fn = metric_modules[metric_name].compute

            no_cond = getattr(metric_modules[metric_name], 'NO_CONDITIONS', False)

            if no_cond:
                result = compute_fn(stats)
            elif conditions_enabled and len(conditions_enabled) > 0:
                result = cond_analyzer.compute_all(compute_fn, stats, enabled=conditions_enabled)
                result["_condition_info"] = cond_analyzer.get_threshold_info()
            else:
                result = compute_fn(stats)

            fname = os.path.join(save_dir, f'{metric_name}.json')
            json.dump(result, open(fname, 'w'), indent=2)
            print(f"  {metric_name}: saved to {fname}")

            # quick summary
            if isinstance(result, dict) and 'overall' in result:
                r0 = result.get('overall', result)
            else:
                r0 = result if isinstance(result, dict) else {}
            for k, v in r0.items():
                if isinstance(v, (int, float)) and v is not None:
                    print(f"    {k}: {v}")
                elif isinstance(v, np.floating):
                    print(f"    {k}: {float(v)}")

        # ---- visualizations (per split, after all metrics) ----
        if visualization_enabled:
            viz_dir = os.path.join(base_out, split)
            for viz_name in visualization_enabled:
                if viz_name == "scatter":
                    from visualization.plots import plot_error_scatter
                    plot_error_scatter(stats, split, out_dir=viz_dir)
                elif viz_name == "histogram":
                    from visualization.plots import plot_histogram
                    plot_histogram(stats, split, out_dir=viz_dir)
                elif viz_name == "epi_winrate":
                    from visualization.plots import plot_epi_winrate
                    plot_epi_winrate(stats, split, out_dir=viz_dir)
                elif viz_name == "calibration_error":
                    from visualization.plots import plot_calibration_error
                    plot_calibration_error(stats, split, out_dir=viz_dir)
                elif viz_name == "calibration_gain":
                    from visualization.plots import plot_calibration_gain
                    plot_calibration_gain(stats, split, out_dir=viz_dir)
                elif viz_name == "threshold_sweep":
                    from visualization.plots import plot_threshold_sweep
                    plot_threshold_sweep(stats, split, out_dir=viz_dir)

    print(f"\nDone! Results in {base_out}/")
