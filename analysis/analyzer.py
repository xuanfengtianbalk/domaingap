#!/usr/bin/env python3
"""Analysis module entry point.

Usage:
    cd analysis && python analyzer.py

Configuration: edit config.yaml
"""

from __future__ import annotations
import os, sys, yaml


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get('mode', 'pairwise')
    if mode == 'pairwise':
        p = cfg['pairwise']
        metrics_cfg = cfg.get('metrics', ['correlation', 'winrate', 'direction'])
        conds_cfg = cfg.get('conditions', [])
        viz_cfg = cfg.get('visualization', [])
        from modes.pairwise import run_pairwise
        run_pairwise(
            uuid_1=p['uuid_1'],
            uuid_2=p['uuid_2'],
            splits=p.get('splits', ['validation']),
            max_samples=p.get('max_samples'),
            batch_size=p.get('batch_size', 1),
            device=p.get('device', 'cuda:0'),
            metrics_enabled=metrics_cfg,
            conditions_enabled=conds_cfg,
            visualization_enabled=viz_cfg,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    main()
