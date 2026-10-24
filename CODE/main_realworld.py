"""Preprocessing used by the current TriGAC implementation.

The legacy experiment loop was removed. This module retains the import name
expected by existing TriGAC runners and only exposes the shared transform.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_multiview_data(
    X_list_original,
    high_dim_threshold: int = 50,
    umap_n_components: int = 20,
    random_state: int = 42,
):
    """Z-score each view, reduce it as configured, then L2-normalize rows.

    Above 50 dimensions: PCA-50 followed by UMAP-20. From 20 through 50
    dimensions: UMAP-20 only. Below 20 dimensions: no reduction. The
    parameters and operation order match the paper's current pipeline.
    """
    processed = []
    for view in X_list_original:
        features = np.asarray(view)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("Each view must be a finite n-by-d matrix")
        values = StandardScaler().fit_transform(features)
        n_features = values.shape[1]
        if n_features > high_dim_threshold:
            from umap import UMAP

            values = PCA(
                n_components=high_dim_threshold,
                random_state=random_state,
            ).fit_transform(values)
            values = UMAP(
                n_components=umap_n_components,
                random_state=random_state,
                n_neighbors=20,
                min_dist=0.1,
                metric="euclidean",
            ).fit_transform(values)
        elif n_features >= umap_n_components:
            from umap import UMAP

            values = UMAP(
                n_components=umap_n_components,
                random_state=random_state,
                n_neighbors=20,
                min_dist=0.1,
                metric="euclidean",
            ).fit_transform(values)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        processed.append(values / norms)
    return processed
