import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import BallTree


class GeneralUtils:
    def __init__(self,n_samples,n_views):
        self.n_samples=n_samples
        self.n_views=n_views


    def _local_regions_search(self, X_list, k):
        n_samples = self.n_samples
        local_regions = []
        for view_idx, X in enumerate(X_list):

            tree = BallTree(X)

            distances, indices = tree.query(X, k)

            neighbors_this_view = {}
            for i in range(n_samples):
                neighbor_indices = indices[i].tolist()
                neighbors_this_view[i] = neighbor_indices

            local_regions.append(neighbors_this_view)
        return local_regions



    def _refined_result_to_metrics(self,refined_result, y):

        n_samples = len(y)
        pred_labels = []
        for node_id in range(n_samples):
            pred_labels.append(refined_result.get(node_id, -1))  
        pred_labels = np.array(pred_labels)

        valid_mask = pred_labels != -1
        y_valid = y[valid_mask]
        pred_valid = pred_labels[valid_mask]
        if len(y_valid) == 0:
            ari = np.nan
            nmi = np.nan
        else:
            ari = adjusted_rand_score(y_valid, pred_valid)
            nmi = normalized_mutual_info_score(y_valid, pred_valid)
        return ari, nmi,pred_labels


