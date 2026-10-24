#!/usr/bin/env python
"""Paper-aligned TriGAC algorithm for the 11 small datasets.

Each of the five query caps is a separate run. Existing experiment results are
never used as outputs or overwritten. The original preprocessing and object-
tree topology are reused; selection, direct anchors, ML rewiring, and feedback
timing are implemented in paper_core.py.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ALGORITHM = "TriGAC-paper-aligned-v1"

from view_weighting import (  # noqa: E402
    MarginalContributionTracker,
    MarginalGroupWeightStrategy,
)


from paper_config import DATASET_CONFIGS

from paper_core import (  # noqa: E402
    PaperAnnotator,
    PaperRefiner,
    PaperSelector,
    component_partitions,
    structural_scores,
)


def encode(value) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False)


def run_one(prepared: Dict, name: str, seed: int, cap: int,
            rounds: int, omega: int, version: str,
            stop_on_perfect: bool = False) -> pd.DataFrame:
    features = prepared["X_list"]
    truth = np.asarray(prepared["y"]).reshape(-1)
    original_graphs = copy.deepcopy(prepared["node_skeletons"])
    n = len(truth)
    m = len(features)
    if not features or any(len(view) != n for view in features):
        raise ValueError("The view dimensions or ground-truth length disagree")
    root_query_bound = m * (m - 1) // 2
    if cap < root_query_bound:
        raise ValueError(
            f"Cap {cap} is below the manuscript's root-processing bound "
            f"{root_query_bound} for {m} views"
        )

    scores, roots = structural_scores(original_graphs)
    selector = PaperSelector(scores, roots)
    annotator = PaperAnnotator(features, truth, scores, omega, cap)
    refiner = PaperRefiner(original_graphs)
    tracker = MarginalContributionTracker(m)
    weighting = MarginalGroupWeightStrategy(n, m, tracker)
    weights = np.ones(m, dtype=float) / m
    nominal_batch = int(math.ceil(n / float(rounds)))

    records: List[Dict] = [dict(
        dataset=name,
        display_name=DATASET_CONFIGS[name].display_name,
        seed=seed,
        max_budget=cap,
        count=0,
        iteration=0,
        selected_num=0,
        processed_num=0,
        ARI=adjusted_rand_score(truth, np.zeros(n, dtype=int)),
        NMI=normalized_mutual_info_score(truth, np.zeros(n, dtype=int)),
        weights_used=encode(weights),
        weights_next=encode(weights),
        root_nodes=encode(roots),
        root_processed=False,
        selected_order="",
        view_clusters="",
        group_contributions="",
        view_contributions="",
        selected_mean_uncertainty=0.0,
        selected_max_uncertainty=0.0,
        evaluated_samples_this_round=0,
        cumulative_evaluated_samples=0,
        T=rounds,
        Omega=omega,
        algorithm=ALGORITHM,
        protocol_sha=version,
        round_runtime_sec=0.0,
        termination_reason="",
    )]

    for iteration in range(1, rounds + 1):
        started = time.perf_counter()
        if not selector.unselected:
            records[-1]["termination_reason"] = "all_objects_selected"
            break
        weights_used = weights.copy()
        pre_query_partitions = component_partitions(
            refiner.refined_graphs if refiner.refined_graphs else original_graphs, n
        )
        if iteration == 1:
            selected = selector.first_round(nominal_batch)
        else:
            selected = selector.later_round(
                nominal_batch, weights_used, features, annotator.windows
            )
        if not selected:
            records[-1]["termination_reason"] = "no_objects_selected"
            break

        round_queries = annotator.annotate(selected, weights_used)
        logged = sum(len(queries) for queries in round_queries.values())
        if logged != annotator.count - int(records[-1]["count"]):
            raise AssertionError("The direct-query log and query count disagree")
        # Include round 1: its pre-update predictions are part of the
        # manuscript's cumulative annotation history.
        tracker.update(pre_query_partitions, round_queries)
        refiner.update_constraints(annotator.neighborhoods, round_queries)
        root_processed = set(roots).issubset(refiner.parents)
        if iteration == 1 and not root_processed:
            raise AssertionError(
                "A final representative agent was not processed in round 1; "
                "WCC correspondence would not be justified"
            )
        refiner.refine(annotator.neighborhoods, weights_used)
        predictions = np.fromiter(
            (refiner.node_final_neigh[node] for node in range(n)),
            dtype=int, count=n,
        )
        ari = float(adjusted_rand_score(truth, predictions))
        nmi = float(normalized_mutual_info_score(truth, predictions))

        # The updated partitions and accumulated feedback determine weights
        # for the *next* interaction round; round 1 itself remains uniform.
        (weights, groups, group_mc, view_mc, _, _) = (
            weighting.update_hierarchical(refiner.labeling_result)
        )
        if stop_on_perfect and ari >= 1.0 - 1e-12:
            reason = "perfect_clustering"
        elif annotator.budget_reached or annotator.count >= cap:
            reason = "budget_reached"
        elif not selector.unselected:
            reason = "all_objects_selected"
        elif iteration == rounds:
            reason = "max_iterations"
        else:
            reason = ""
        records.append(dict(
            dataset=name,
            display_name=DATASET_CONFIGS[name].display_name,
            seed=seed,
            max_budget=cap,
            count=annotator.count,
            iteration=iteration,
            selected_num=n - len(selector.unselected),
            processed_num=len(refiner.parents),
            ARI=ari,
            NMI=nmi,
            weights_used=encode(weights_used),
            weights_next=encode(weights),
            root_nodes=encode(roots),
            root_processed=root_processed,
            selected_order=encode(selected) if iteration == 1 else "",
            view_clusters=encode(groups),
            group_contributions=encode(group_mc),
            view_contributions=encode(view_mc),
            selected_mean_uncertainty=selector.last_mean_uncertainty,
            selected_max_uncertainty=selector.last_max_uncertainty,
            evaluated_samples_this_round=tracker.last_round_samples,
            cumulative_evaluated_samples=len(tracker.samples),
            T=rounds,
            Omega=omega,
            algorithm=ALGORITHM,
            protocol_sha=version,
            round_runtime_sec=time.perf_counter() - started,
            termination_reason=reason,
        ))
        print(
            f"{name} seed={seed} cap={cap} round={iteration}/{rounds} "
            f"q={annotator.count} processed={len(refiner.parents)} "
            f"ARI={ari:.6f} roots_ok={root_processed}",
            flush=True,
        )
        if reason in ("budget_reached", "all_objects_selected", "perfect_clustering"):
            break
    return pd.DataFrame.from_records(records)


def verify_result(path: Path, name: str, seed: int, cap: int,
                  rounds: int, omega: int, version: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty result: {path}")
    last = frame.iloc[-1]
    expected = (
        str(last["dataset"]) == name,
        int(last["seed"]) == seed,
        int(last["max_budget"]) == cap,
        int(last["T"]) == rounds,
        int(last["Omega"]) == omega,
        str(last["algorithm"]) == ALGORITHM,
        str(last["protocol_sha"]) == version,
        0 <= int(last["count"]) <= cap,
        str(last["root_processed"]).lower() == "true",
        math.isfinite(float(last["ARI"])),
        math.isfinite(float(last["NMI"])),
    )
    if not all(expected):
        raise ValueError(f"Existing result does not match the current protocol: {path}")
    return frame
