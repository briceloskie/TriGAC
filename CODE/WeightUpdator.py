import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from ViewSkeletonBuilder import ViewSkeletonBuilder


class WeightUpdator:
    def __init__(self, n_samples, n_views):
        self.n_samples = n_samples
        self.n_views = n_views

    def _calculate_view_similarity_matrix(self, labeling_result, method='nmi'):
        """
        Calculate similarity matrix between views

        :param labeling_result: list of dict, node-to-neighborhood-label mapping for each view [{node_id: label}, ...]
        :param method: str, similarity calculation method 'nmi' or 'ari'
        :return: np.array, similarity matrix (n_views, n_views), symmetric matrix with diagonal values of 1
        """
        n_views = len(labeling_result)
        similarity_matrix = np.zeros((n_views, n_views))

        # Extract label arrays for all views
        view_labels_list = []
        for view_idx in range(n_views):
            view_node2label = labeling_result[view_idx]
            view_labels = np.array([view_node2label.get(node_id, -1) for node_id in range(self.n_samples)])
            view_labels_list.append(view_labels)

        # Calculate similarity for each pair of views
        for i in range(n_views):
            for j in range(i, n_views):
                if i == j:
                    # Diagonal: similarity of view with itself is 1
                    similarity_matrix[i, j] = 1.0
                else:
                    # Get labels for two views
                    labels_i = view_labels_list[i]
                    labels_j = view_labels_list[j]

                    # Get valid nodes present in both views
                    valid_mask = (labels_i != -1) & (labels_j != -1)

                    if not np.any(valid_mask):
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
                        continue

                    labels_i_valid = labels_i[valid_mask]
                    labels_j_valid = labels_j[valid_mask]

                    # Check if there are enough unique labels
                    if len(np.unique(labels_i_valid)) < 2 or len(np.unique(labels_j_valid)) < 2:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
                        continue

                    # Calculate similarity
                    try:
                        if method == 'nmi':
                            sim = normalized_mutual_info_score(labels_i_valid, labels_j_valid)
                        elif method == 'ari':
                            sim = adjusted_rand_score(labels_i_valid, labels_j_valid)
                        else:
                            raise ValueError(f"Unsupported similarity method: {method}")

                        # Symmetric matrix
                        similarity_matrix[i, j] = max(sim, 0.0)
                        similarity_matrix[j, i] = max(sim, 0.0)
                    except Exception as e:
                        print(f"⚠️ Similarity calculation failed for view {i} and view {j}: {e}")
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0

        return similarity_matrix

    def _calculate_view_quality(self, labeling_result, node_final_neigh, method='nmi'):
        """
        Calculate quality score q_v for individual view: structural similarity between view skeleton and fused skeleton

        :param labeling_result: list of dict, node-to-neighborhood-label mapping for each view
        :param node_final_neigh: dict, fused node-to-neighborhood-label mapping
        :param method: str, similarity calculation method 'nmi' or 'ari'
        :return: np.array, quality score for each view (n_views,)
        """
        n_views = len(labeling_result)
        view_qualities = np.zeros(n_views)

        # Extract label array for fused result
        fused_labels = np.array([node_final_neigh.get(node_id, -1) for node_id in range(self.n_samples)])
        valid_mask = fused_labels != -1

        if not np.any(valid_mask):
            print("⚠️ Warning: No valid labels in fused result, returning uniform quality")
            return np.ones(n_views) / n_views

        fused_labels_valid = fused_labels[valid_mask]

        # Calculate similarity between each view and fused result as quality score
        for view_idx in range(n_views):
            view_node2label = labeling_result[view_idx]
            view_labels = np.array([view_node2label.get(node_id, -1) for node_id in range(self.n_samples)])
            view_labels_valid = view_labels[valid_mask]

            # Filter nodes without labels in the view
            view_valid_mask = view_labels_valid != -1
            if not np.any(view_valid_mask):
                view_qualities[view_idx] = 0.0
                continue

            # Get nodes valid in both view and fused result
            common_valid_mask = view_valid_mask
            fused_common = fused_labels_valid[common_valid_mask]
            view_common = view_labels_valid[common_valid_mask]

            # Check if there are enough unique labels
            if len(np.unique(fused_common)) < 2 or len(np.unique(view_common)) < 2:
                view_qualities[view_idx] = 0.0
                continue

            # Calculate similarity as quality score
            try:
                if method == 'nmi':
                    quality = normalized_mutual_info_score(fused_common, view_common)
                elif method == 'ari':
                    quality = adjusted_rand_score(fused_common, view_common)
                else:
                    raise ValueError(f"Unsupported similarity method: {method}")

                view_qualities[view_idx] = max(quality, 0.0)
            except Exception as e:
                print(f"⚠️ Quality calculation failed for view {view_idx}: {e}")
                view_qualities[view_idx] = 0.0

        return view_qualities

    def update_weights(self, labeling_result, node_final_neigh, method='nmi', n_clusters=None):
        """
        Hierarchical quality-driven view weight update (three-step strategy)

        Step 1: Cluster views based on similarity matrix to get K clusters
        Step 2: Assign total weight quota to each cluster (proportional to overall cluster quality)
        Step 3: Intra-cluster weight allocation (proportional to individual view quality)

        :param labeling_result: list of dict, node-to-neighborhood-label mapping for each view
        :param node_final_neigh: dict, fused node-to-neighborhood-label mapping
        :param method: str, similarity calculation method 'nmi' or 'ari'
        :param n_clusters: int, expected number of clusters (None for automatic determination)
        :return: np.array, updated view weights (n_views,), weights sum to 1
        """
        n_views = len(labeling_result)

        # ===================== Step 1: View Clustering =====================
        # print("\n📌 Step 1: Clustering based on view similarity matrix")

        # Calculate similarity matrix between views
        view_similarity_matrix = self._calculate_view_similarity_matrix(labeling_result, method=method)
        # print(f"View similarity matrix:\n{np.round(view_similarity_matrix, 4)}")

        # Use ViewSkeletonBuilder to build view skeleton and divide into clusters
        viewbuilder = ViewSkeletonBuilder()
        view_skeleton = viewbuilder.build(view_similarity_matrix)
        view_clusters = viewbuilder.clusters_division(view_skeleton)

        K = len(view_clusters)
        # print(f"✅ View clustering completed: {n_views} views → {K} clusters")
        # for k, cluster in enumerate(view_clusters):
        #     print(f"  Cluster {k}: views {cluster}")

        # ===================== Step 2: Calculate View Quality =====================
        # print("\n📌 Step 2: Calculate quality score for each view")
        view_qualities = self._calculate_view_quality(labeling_result, node_final_neigh, method=method)
        # print(f"View quality scores: {[round(q, 4) for q in view_qualities]}")

        # ===================== Step 3: Cluster-level Quota Allocation =====================
        # print("\n📌 Step 3: Allocate total weight quota based on cluster quality")
        cluster_weights = np.zeros(K)

        for k, cluster in enumerate(view_clusters):
            # Overall cluster quality = average quality of views in the cluster
            cluster_quality = np.mean([view_qualities[v] for v in cluster])
            cluster_weights[k] = cluster_quality

        # Normalize: ensure all cluster quotas sum to 1
        total_cluster_weight = np.sum(cluster_weights)
        if total_cluster_weight < 1e-12:
            # All cluster qualities are 0, allocate uniformly
            cluster_weights = np.ones(K) / K
        else:
            cluster_weights = cluster_weights / total_cluster_weight

        # print(f"Cluster weight quotas: {[round(w, 4) for w in cluster_weights]}")

        # ===================== Step 4: Intra-cluster Weight Allocation =====================
        # print("\n📌 Step 4: Intra-cluster weight allocation (based on individual quality proportion)")
        final_weights = np.zeros(n_views)

        for k, cluster in enumerate(view_clusters):
            # Quality list for views in the cluster
            cluster_qualities = np.array([view_qualities[v] for v in cluster])
            total_quality = np.sum(cluster_qualities)

            if total_quality < 1e-12:
                # All view qualities in cluster are 0, evenly distribute cluster quota
                for v in cluster:
                    final_weights[v] = cluster_weights[k] / len(cluster)
            else:
                # Allocate cluster quota proportional to quality
                for idx, v in enumerate(cluster):
                    final_weights[v] = cluster_weights[k] * (cluster_qualities[idx] / total_quality)

        # print(f"✅ Final view weights (sum={np.sum(final_weights):.4f}): {[round(w, 4) for w in final_weights]}")

        return final_weights
