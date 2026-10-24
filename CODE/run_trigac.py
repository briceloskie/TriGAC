#!/usr/bin/env python3
"""Run the paper-aligned TriGAC on the 11 small multi-view datasets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.data_loader import load_multiview_dataset
from NodeSkeletonBuilder import NodeSkeletonBuilder
from paper_config import DATASET_CONFIGS, SMALL_DATASETS
from paper_engine import ALGORITHM, run_one, verify_result


HERE = Path(__file__).resolve().parent
PROTOCOL_FILES = (
    "run_trigac.py", "paper_config.py", "paper_engine.py", "paper_core.py",
    "view_weighting.py", "NodeSkeletonBuilder.py", "ViewSkeletonBuilder.py",
    "WeightUpdator.py", "main_realworld.py", "datasets/data_loader.py",
)


def dataset_path(name: str, data_dir: Path) -> Path:
    return data_dir / f"{DATASET_CONFIGS[name].loader_name}.mat"


def protocol_hash(path: Path) -> str:
    """Invalidate an existing trajectory if code or the input file changes."""
    digest = hashlib.sha256()
    for relative in PROTOCOL_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((HERE / relative).read_bytes())
    stat = path.stat()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def prepare_dataset(path: Path, seed: int, pca_dim: int, umap_dim: int) -> dict:
    from main_realworld import preprocess_multiview_data

    original, labels = load_multiview_dataset(str(path))
    views = preprocess_multiview_data(
        original, high_dim_threshold=pca_dim,
        umap_n_components=umap_dim, random_state=seed,
    )
    del original
    gc.collect()
    graphs, centralities = NodeSkeletonBuilder().build(views)
    return {
        "X_list": views,
        "y": np.asarray(labels).reshape(-1),
        "node_skeletons": graphs,
        "centralities_dicts": centralities,
    }


def result_path(root: Path, name: str, seed: int, cap: int) -> Path:
    return root / "raw" / f"{name}__{ALGORITHM}__seed{seed}__budget{cap}.csv"


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", choices=SMALL_DATASETS)
    target.add_argument("--all", action="store_true", help="Run all 11 small datasets")
    parser.add_argument("--budget", type=int, help="One cap; requires --dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t", type=int, default=25)
    parser.add_argument("--omega", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--umap-dim", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=HERE / "datasets")
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.budget is not None and (args.all or args.budget < 3):
        parser.error("--budget must be at least 3 and requires --dataset")
    if args.seed < 0 or min(args.t, args.omega, args.pca_dim, args.umap_dim) < 1:
        parser.error("Seed must be nonnegative; dimensions, T, Omega must be positive")

    names = SMALL_DATASETS if args.all else (args.dataset,)
    for name in names:
        source = dataset_path(name, args.data_dir)
        if not source.is_file():
            raise FileNotFoundError(source)
        caps = (args.budget,) if args.budget is not None else DATASET_CONFIGS[name].budgets
        print(f"INPUT {name}: {source}; caps={caps}", flush=True)
    if args.validate_only:
        print("Validated all requested inputs; no experiment started", flush=True)
        return

    run_root = args.output_root / (
        f"seed{args.seed}_T{args.t}_Omega{args.omega}"
        f"_PCA{args.pca_dim}_UMAP{args.umap_dim}"
    )
    summary = []
    for name in names:
        source = dataset_path(name, args.data_dir)
        version = protocol_hash(source)
        caps = (args.budget,) if args.budget is not None else DATASET_CONFIGS[name].budgets
        prepared = None
        for stage, cap in enumerate(caps, start=1):
            output = result_path(run_root, name, args.seed, cap)
            if output.is_file():
                frame = verify_result(output, name, args.seed, cap,
                                      args.t, args.omega, version)
                print(f"SKIP validated {output}", flush=True)
            else:
                if prepared is None:
                    prepared = prepare_dataset(
                        source, args.seed, args.pca_dim, args.umap_dim
                    )
                print(f"RUN {name} stage={stage} cap={cap}", flush=True)
                frame = run_one(prepared, name, args.seed, cap,
                                args.t, args.omega, version)
                frame["stage"] = stage
                frame["pca_dim"] = args.pca_dim
                frame["umap_dim"] = args.umap_dim
                save_frame(frame, output)
                verify_result(output, name, args.seed, cap,
                              args.t, args.omega, version)
            last = frame.iloc[-1]
            summary.append({
                "dataset": DATASET_CONFIGS[name].display_name,
                "dataset_key": name,
                "stage": stage,
                "cons_cap": cap,
                "actual_queries": int(last["count"]),
                "ARI": float(last["ARI"]),
                "NMI": float(last["NMI"]),
                "seed": args.seed,
            })
            print(f"DONE {name} cap={cap} actual={int(last['count'])} "
                  f"ARI={float(last['ARI']):.6f}", flush=True)
        del prepared
        gc.collect()
        save_frame(pd.DataFrame(summary), run_root / "summary.csv")
    print("PAPER-ALIGNED TRIGAC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
