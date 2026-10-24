"""View-graph marginal-contribution weighting for paper-aligned TriGAC.

The view graph is retained solely to define redundancy groups. Starting from
the exact Uniform 1/m prior, a group is upweighted only when removing it makes
prediction of accumulated direct constraints worse.
"""

from typing import List, Sequence, Tuple

import numpy as np


class MarginalContributionTracker:
    """Store pre-query predictions, with equal weight for every sample."""

    def __init__(self, n_views: int):
        self.n_views = int(n_views)
        # Each entry is (predictions[q, view], oracle_answers[q]).
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.last_round_samples = 0

    def update(self, pre_query_partitions, round_queries) -> None:
        added = 0
        for queries in round_queries.values():
            if not queries:
                continue
            predictions = np.zeros((len(queries), self.n_views), dtype=float)
            answers = np.zeros(len(queries), dtype=float)
            for query_idx, (node, reference, relation) in enumerate(queries):
                answers[query_idx] = float(relation)
                for view_idx, labels in enumerate(pre_query_partitions):
                    predictions[query_idx, view_idx] = float(
                        labels[node] == labels[reference]
                    )
            self.samples.append((predictions, answers))
            added += 1
        self.last_round_samples = added

    def group_contributions(self, groups: Sequence[Sequence[int]]) -> np.ndarray:
        """Delta_l = loss_without_group_l - loss_with_all_views.

        Loss is averaged inside each selected sample first, so a sample that
        generates many cannot-link queries does not dominate the statistic.
        """
        contributions = np.zeros(len(groups), dtype=float)
        if not self.samples:
            return contributions

        per_group_sample_delta: List[List[float]] = [
            [] for _ in range(len(groups))
        ]
        all_views = np.arange(self.n_views)
        for predictions, answers in self.samples:
            uniform_prediction = predictions.mean(axis=1)
            uniform_loss = float(np.mean((uniform_prediction - answers) ** 2))
            for group_idx, group in enumerate(groups):
                group_set = set(map(int, group))
                retained = np.asarray(
                    [view for view in all_views if int(view) not in group_set],
                    dtype=int,
                )
                if retained.size == 0:
                    # A single group containing every view has no meaningful
                    # leave-group-out comparison and therefore no adjustment.
                    delta = 0.0
                else:
                    without_prediction = predictions[:, retained].mean(axis=1)
                    without_loss = float(
                        np.mean((without_prediction - answers) ** 2)
                    )
                    delta = without_loss - uniform_loss
                per_group_sample_delta[group_idx].append(delta)

        for group_idx, values in enumerate(per_group_sample_delta):
            if values:
                contributions[group_idx] = float(np.mean(values))
        return contributions

    @staticmethod
    def _relative_marginal(loss_without: float, loss_with: float) -> float:
        denominator = max(float(loss_without), float(loss_with))
        if denominator < 1e-12:
            return 0.0
        return max(0.0, (float(loss_without) - float(loss_with)) / denominator)

    def hierarchical_relative_contributions(
        self,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Relative marginal contributions at group and within-group levels."""
        group_mc = np.zeros(len(groups), dtype=float)
        view_mc = np.zeros(self.n_views, dtype=float)
        if not self.samples:
            return group_mc, view_mc

        all_losses: List[float] = []
        without_group_losses: List[List[float]] = [
            [] for _ in range(len(groups))
        ]
        group_losses: List[List[float]] = [[] for _ in range(len(groups))]
        without_view_losses: List[List[float]] = [
            [] for _ in range(self.n_views)
        ]
        all_views = np.arange(self.n_views)

        for predictions, answers in self.samples:
            all_prediction = predictions.mean(axis=1)
            all_losses.append(float(np.mean((all_prediction - answers) ** 2)))

            for group_idx, group in enumerate(groups):
                group = list(map(int, group))
                group_set = set(group)
                retained = np.asarray(
                    [view for view in all_views if int(view) not in group_set],
                    dtype=int,
                )
                if retained.size == 0:
                    without_group_losses[group_idx].append(np.nan)
                else:
                    prediction_without_group = predictions[:, retained].mean(axis=1)
                    without_group_losses[group_idx].append(
                        float(np.mean((prediction_without_group - answers) ** 2))
                    )

                group_prediction = predictions[:, group].mean(axis=1)
                group_losses[group_idx].append(
                    float(np.mean((group_prediction - answers) ** 2))
                )
                for view in group:
                    retained_in_group = [candidate for candidate in group if candidate != view]
                    if not retained_in_group:
                        without_view_losses[view].append(np.nan)
                    else:
                        prediction_without_view = predictions[:, retained_in_group].mean(axis=1)
                        without_view_losses[view].append(
                            float(np.mean((prediction_without_view - answers) ** 2))
                        )

        loss_all = float(np.mean(all_losses))
        for group_idx, group in enumerate(groups):
            valid_group_without = np.asarray(
                without_group_losses[group_idx], dtype=float
            )
            valid_group_without = valid_group_without[
                np.isfinite(valid_group_without)
            ]
            if valid_group_without.size:
                group_mc[group_idx] = self._relative_marginal(
                    float(np.mean(valid_group_without)),
                    loss_all,
                )

            loss_group = float(np.mean(group_losses[group_idx]))
            for view in group:
                valid_view_without = np.asarray(
                    without_view_losses[int(view)], dtype=float
                )
                valid_view_without = valid_view_without[
                    np.isfinite(valid_view_without)
                ]
                if valid_view_without.size:
                    view_mc[int(view)] = self._relative_marginal(
                        float(np.mean(valid_view_without)),
                        loss_group,
                    )
        return group_mc, view_mc


class MarginalGroupWeightStrategy:
    """Uniform-anchored group weighting without a tunable parameter."""

    def __init__(self, n_samples: int, n_views: int, tracker):
        from WeightUpdator import WeightUpdator

        self.n_samples = int(n_samples)
        self.n_views = int(n_views)
        self.tracker = tracker
        self.updator = WeightUpdator(n_samples, n_views)

    def update(self, labeling_result):
        from ViewSkeletonBuilder import ViewSkeletonBuilder

        similarity = self.updator._calculate_view_similarity_matrix(
            labeling_result, method="nmi"
        )
        builder = ViewSkeletonBuilder()
        skeleton = builder.build(similarity)
        groups = [
            list(map(int, group))
            for group in builder.clusters_division(skeleton)
        ]
        if not groups:
            weights = np.ones(self.n_views, dtype=float) / self.n_views
            return weights, [], np.array([], dtype=float), similarity, np.array([])

        covered = sorted(view for group in groups for view in group)
        if covered != list(range(self.n_views)):
            raise ValueError(f"View groups are not a partition: {groups}")

        contributions = self.tracker.group_contributions(groups)
        positive_contributions = np.maximum(contributions, 0.0)
        group_sizes = np.asarray([len(group) for group in groups], dtype=float)

        # beta_l proportional to |C_l| * (1 + [Delta_l]_+).
        # When every Delta_l <= 0, beta_l=|C_l|/m and each view receives 1/m.
        group_scores = group_sizes * (1.0 + positive_contributions)
        group_weights = group_scores / group_scores.sum()
        weights = np.zeros(self.n_views, dtype=float)
        for group_idx, group in enumerate(groups):
            weights[group] = group_weights[group_idx] / len(group)
        weights /= weights.sum()
        return (
            weights,
            groups,
            contributions,
            np.asarray(similarity, dtype=float),
            group_weights,
        )

    def update_hierarchical(self, labeling_result):
        """Group-level and within-group relative marginal contributions."""
        from ViewSkeletonBuilder import ViewSkeletonBuilder

        similarity = self.updator._calculate_view_similarity_matrix(
            labeling_result, method="nmi"
        )
        builder = ViewSkeletonBuilder()
        skeleton = builder.build(similarity)
        groups = [
            list(map(int, group))
            for group in builder.clusters_division(skeleton)
        ]
        if not groups:
            weights = np.ones(self.n_views, dtype=float) / self.n_views
            return (
                weights,
                [],
                np.array([], dtype=float),
                np.zeros(self.n_views, dtype=float),
                similarity,
                np.array([], dtype=float),
            )

        covered = sorted(view for group in groups for view in group)
        if covered != list(range(self.n_views)):
            raise ValueError(f"View groups are not a partition: {groups}")

        group_mc, view_mc = self.tracker.hierarchical_relative_contributions(groups)
        scores = np.ones(self.n_views, dtype=float)
        for group_idx, group in enumerate(groups):
            scores[group] = (1.0 + group_mc[group_idx]) * (1.0 + view_mc[group])
        weights = scores / scores.sum()
        group_weights = np.asarray(
            [float(weights[group].sum()) for group in groups], dtype=float
        )
        return (
            weights,
            groups,
            group_mc,
            view_mc,
            np.asarray(similarity, dtype=float),
            group_weights,
        )
