"""Paper-aligned selection, direct feedback, and object-graph refinement.

The initialized object-tree topology and the view-weighting implementation
are reused from the existing TriGAC project. This module changes the parts
that differed from the October 2026 manuscript.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


Query = Tuple[int, int, int]  # object, queried anchor, ML=1 or CL=0


def structural_scores(initial_graphs: Sequence[nx.DiGraph]) -> Tuple[np.ndarray, List[int]]:
    """Return r=1-p for non-roots and r=1 for each final root.

    The graph builder stores p as ``similarity`` on each outgoing edge. The
    edge's distance is used only as a checked fallback for older caches.
    """
    if not initial_graphs:
        raise ValueError("At least one object graph is required")
    n = len(initial_graphs[0])
    scores = np.zeros((len(initial_graphs), n), dtype=float)
    roots = []
    for view, graph in enumerate(initial_graphs):
        if set(graph.nodes) != set(range(n)):
            raise ValueError("Each object graph must contain nodes 0,...,n-1")
        view_roots = [int(node) for node in graph if graph.out_degree(node) == 0]
        if len(view_roots) != 1 or any(graph.out_degree(node) > 1 for node in graph):
            raise ValueError("Each initialized object graph must be a rooted out-tree")
        roots.append(view_roots[0])
        for node in range(n):
            outgoing = list(graph.out_edges(node, data=True))
            if not outgoing:
                scores[view, node] = 1.0
                continue
            edge = outgoing[0][2]
            if "similarity" in edge:
                probability = float(edge["similarity"])
            else:
                distance = float(edge["distance"])
                probability = 1.0 / (1.0 + distance + 1e-12)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("An object-edge probability is outside [0,1]")
            scores[view, node] = 1.0 - probability
            graph.nodes[node]["centrality"] = float(scores[view, node])
        graph.nodes[roots[-1]]["centrality"] = 1.0
    return scores, roots


def component_partitions(graphs: Sequence[nx.DiGraph], n: int) -> List[np.ndarray]:
    """Get the complete per-view WCC labels before feedback in a round."""
    partitions = []
    for graph in graphs:
        labels = np.full(n, -1, dtype=int)
        for component_id, component in enumerate(nx.weakly_connected_components(graph)):
            labels[list(component)] = component_id
        if np.any(labels < 0):
            raise AssertionError("Incomplete pre-query object-graph partition")
        partitions.append(labels)
    return partitions


class PaperSelector:
    """Root-first cold start; later global WCC disagreement with R tie-break."""

    def __init__(self, scores: np.ndarray, roots: Sequence[int]):
        self.scores = np.asarray(scores, dtype=float)
        self.n_views, self.n = self.scores.shape
        self.roots = list(dict.fromkeys(int(root) for root in roots))
        self.unselected = set(range(self.n))
        self.last_mean_uncertainty = 0.0
        self.last_max_uncertainty = 0.0

    def _representativeness(self, candidates: Sequence[int], weights: np.ndarray) -> np.ndarray:
        ids = np.asarray(candidates, dtype=int)
        values = self.scores[:, ids]
        denominators = np.max(values, axis=1)
        denominators[denominators <= 1e-12] = 1.0
        return np.sum(weights[:, None] * values / denominators[:, None], axis=0)

    def first_round(self, nominal_batch: int) -> List[int]:
        candidates = sorted(self.unselected)
        weights = np.ones(self.n_views, dtype=float) / self.n_views
        priority = self._representativeness(candidates, weights)
        ranking = sorted(range(len(candidates)), key=lambda i: (-priority[i], candidates[i]))
        top = [candidates[i] for i in ranking[:nominal_batch]]
        # The manuscript takes Top_b union all final roots, then processes the
        # roots first. The union can be larger than the nominal batch.
        root_set = set(self.roots)
        root_order = sorted(root_set, key=lambda node: (-priority[node], node))
        selected = root_order + [node for node in top if node not in root_set]
        self.unselected.difference_update(selected)
        return selected

    def later_round(
        self,
        nominal_batch: int,
        weights: Sequence[float],
        views: Sequence[np.ndarray],
        windows: Sequence[Sequence[Sequence[int]]],
    ) -> List[int]:
        candidates = sorted(self.unselected)
        if not candidates:
            return []
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.n_views,) or not np.isfinite(weights).all():
            raise ValueError("Invalid view-confidence vector")
        weights = weights / weights.sum()
        priority = self._representativeness(candidates, weights)
        predictions = np.empty((len(candidates), self.n_views), dtype=int)
        for view, features in enumerate(views):
            reference_nodes = []
            reference_wccs = []
            for wcc, window in enumerate(windows[view]):
                reference_nodes.extend(map(int, window))
                reference_wccs.extend([wcc] * len(window))
            if not reference_nodes:
                raise ValueError("No constraint-WCC representative windows")
            nearest, _ = pairwise_distances_argmin_min(
                np.asarray(features)[np.asarray(candidates, dtype=int)],
                np.asarray(features)[np.asarray(reference_nodes, dtype=int)],
                metric="euclidean",
            )
            predictions[:, view] = np.asarray(reference_wccs, dtype=int)[nearest]
        uncertainty = np.zeros(len(candidates), dtype=float)
        for row, predicted in enumerate(predictions):
            support = np.bincount(predicted, weights=weights)
            uncertainty[row] = 1.0 - np.max(support)
        ranking = sorted(
            range(len(candidates)),
            key=lambda i: (-uncertainty[i], -priority[i], candidates[i]),
        )
        chosen = ranking[:nominal_batch]
        selected = [candidates[i] for i in chosen]
        self.unselected.difference_update(selected)
        self.last_mean_uncertainty = float(np.mean(uncertainty[chosen]))
        self.last_max_uncertainty = float(np.max(uncertainty[chosen]))
        return selected


class PaperAnnotator:
    """Sequential direct queries using one closest fused-distance anchor/WCC."""

    def __init__(self, views: Sequence[np.ndarray], labels: np.ndarray,
                 scores: np.ndarray, omega: int, cap: int):
        self.views = [np.asarray(view) for view in views]
        self.labels = np.asarray(labels)
        self.scores = np.asarray(scores, dtype=float)
        self.omega = int(omega)
        self.cap = int(cap)
        self.count = 0
        self.budget_reached = False
        self.neighborhoods: List[List[int]] = []
        self.window_heaps: List[List[List[Tuple[float, int, int]]]] = [
            [] for _ in self.views
        ]
        self.windows: List[List[List[int]]] = [[] for _ in self.views]
        if self.omega <= 0 or self.cap <= 0:
            raise ValueError("Omega and the query cap must be positive")

    def _window_entry(self, view: int, node: int) -> Tuple[float, int, int]:
        return (float(self.scores[view, node]), -int(node), int(node))

    def _add_to_window(self, view: int, wcc: int, node: int) -> None:
        heap = self.window_heaps[view][wcc]
        entry = self._window_entry(view, node)
        if len(heap) < self.omega:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
        self.windows[view][wcc] = [item[2] for item in heap]

    def _new_wcc(self, node: int) -> None:
        wcc = len(self.neighborhoods)
        self.neighborhoods.append([node])
        for view in range(len(self.views)):
            self.window_heaps[view].append([])
            self.windows[view].append([])
            self._add_to_window(view, wcc, node)

    def _ordered_anchors(self, node: int, weights: np.ndarray) -> List[Tuple[int, int, float]]:
        references = []
        reference_wcc = []
        for wcc in range(len(self.neighborhoods)):
            union = set()
            for view in range(len(self.views)):
                union.update(self.windows[view][wcc])
            for reference in sorted(union):
                references.append(reference)
                reference_wcc.append(wcc)
        refs = np.asarray(references, dtype=int)
        if refs.size == 0:
            return []
        fused = np.zeros(refs.size, dtype=float)
        for view, features in enumerate(self.views):
            distances = np.linalg.norm(features[refs] - features[node], axis=1)
            spread = float(np.max(distances) - np.min(distances))
            normalized = ((distances - np.min(distances)) / spread
                          if spread > 1e-12 else np.zeros_like(distances))
            fused += float(weights[view]) * normalized
        best = {}
        for index, wcc in enumerate(reference_wcc):
            candidate = (float(fused[index]), int(refs[index]))
            if wcc not in best or candidate < best[wcc]:
                best[wcc] = candidate
        return [
            (wcc, best[wcc][1], best[wcc][0])
            for wcc in sorted(best, key=lambda j: (best[j][0], j))
        ]

    def annotate(self, selected: Sequence[int], weights: Sequence[float]) -> Dict[int, List[Query]]:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        round_queries: Dict[int, List[Query]] = {}
        for node in map(int, selected):
            if not self.neighborhoods:
                self._new_wcc(node)
                continue
            if self.count >= self.cap:
                self.budget_reached = True
                break
            matched = False
            anchors = self._ordered_anchors(node, weights)
            for wcc, anchor, _ in anchors:
                if self.count >= self.cap:
                    self.budget_reached = True
                    break
                relation = int(self.labels[node] == self.labels[anchor])
                self.count += 1
                round_queries.setdefault(node, []).append((node, anchor, relation))
                if relation:
                    self.neighborhoods[wcc].append(node)
                    for view in range(len(self.views)):
                        self._add_to_window(view, wcc, node)
                    matched = True
                    break
            if self.budget_reached:
                break
            if not matched:
                self._new_wcc(node)
        return round_queries


class PaperRefiner:
    """Rewrite object-graph outgoing edges to match the constraint graph."""

    def __init__(self, initial_graphs: Sequence[nx.DiGraph]):
        self.initial_graphs = initial_graphs
        self.n = len(initial_graphs[0])
        self.parents: Dict[int, Optional[int]] = {}
        self.labeling_result: List[Dict[int, int]] = []
        self.node_final_neigh: Dict[int, int] = {}
        self.refined_graphs: List[nx.DiGraph] = []

    def update_constraints(self, neighborhoods: Sequence[Sequence[int]],
                           round_queries: Dict[int, List[Query]]) -> None:
        processed = {int(node) for group in neighborhoods for node in group}
        for node in sorted(processed - set(self.parents)):
            ml = [(int(reference), int(relation))
                  for _, reference, relation in round_queries.get(node, [])
                  if int(relation) == 1]
            if len(ml) > 1:
                raise AssertionError("One object has multiple outgoing ML edges")
            self.parents[node] = ml[0][0] if ml else None
        if set(self.parents) != processed:
            raise AssertionError("Constraint nodes and oracle neighborhoods disagree")
        for node, parent in self.parents.items():
            if parent is not None and parent not in processed:
                raise AssertionError("An ML anchor is missing from the constraint graph")

    def refine(self, neighborhoods: Sequence[Sequence[int]],
               weights: Sequence[float]) -> List[nx.DiGraph]:
        node_to_wcc = {}
        for wcc, group in enumerate(neighborhoods):
            for node in group:
                if node in node_to_wcc:
                    raise AssertionError("Constraint WCCs overlap")
                node_to_wcc[int(node)] = wcc
        if node_to_wcc.keys() != self.parents.keys():
            raise AssertionError("Constraint graph and WCCs contain different nodes")

        self.refined_graphs = []
        self.labeling_result = []
        for original in self.initial_graphs:
            graph = original.copy()
            for node, parent in self.parents.items():
                graph.remove_edges_from(list(graph.out_edges(node)))
                if parent is not None:
                    graph.add_edge(node, parent, constraint_ml=True)
            if any(graph.out_degree(node) > 1 for node in graph):
                raise AssertionError("Refinement violated out-degree <= 1")
            per_view = {}
            for component in nx.weakly_connected_components(graph):
                anchors = {node_to_wcc[node] for node in component if node in node_to_wcc}
                if len(anchors) != 1:
                    raise AssertionError(
                        "WCC correspondence failed; check root-first processing "
                        f"and the budget condition (anchor WCCs={sorted(anchors)})"
                    )
                canonical = next(iter(anchors))
                per_view.update({int(node): canonical for node in component})
            if len(per_view) != self.n:
                raise AssertionError("Incomplete refined object-graph partition")
            self.refined_graphs.append(graph)
            self.labeling_result.append(per_view)

        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        self.node_final_neigh = {}
        for node in range(self.n):
            votes = np.bincount(
                [labels[node] for labels in self.labeling_result],
                weights=weights,
                minlength=len(neighborhoods),
            )
            self.node_final_neigh[node] = int(np.argmax(votes))
        return self.refined_graphs
