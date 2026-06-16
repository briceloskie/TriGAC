import numpy as np
import heapq

class HumanAnnotator:
    def __init__(self,X_list, y,window_size, max_budget=None):
        self.neighborhoods = []
        self.windows=[]
        self.X_list=X_list
        self.y=y
        self.window_size=window_size
        self.count=0
        self.n_views = len(X_list)
        self.window_heaps = []
        self.max_budget = max_budget
        self.budget_reached = False

    def _extract_centrality_from_skeletons(self, node_skeletons):
        centrality_maps = []
        for view_idx, skeleton in enumerate(node_skeletons):
            centrality_map = {}
            for node in skeleton.nodes():
                if 'centrality' in skeleton.nodes[node]:
                    centrality_val = skeleton.nodes[node]['centrality']
                    centrality_map[node] = centrality_val
            centrality_maps.append(centrality_map)
        return centrality_maps

    def _pairwise_contraint_annotation(self, node_skeletons, selected_nodes, view_weights):
        centrality_maps = self._extract_centrality_from_skeletons(node_skeletons)

        for node in selected_nodes:
            if self.budget_reached:
                print(f"   ⚠️ Maximum budget reached ({self.max_budget}), skipping annotation for node {node}")
                continue

            if self.neighborhoods == []:
                first_node = selected_nodes[0]
                self.windows = [[] for _ in range(self.n_views)]
                self.window_heaps = [[] for _ in range(self.n_views)]
                self.neighborhoods = [[selected_nodes[0]]]
                for view_idx in range(self.n_views):
                    self.windows[view_idx].append([first_node])
                    centrality_val = centrality_maps[view_idx].get(first_node, 0.0)
                    self.window_heaps[view_idx].append([(centrality_val, first_node)])
                continue

            annotation_order = self._connection_cal(node, view_weights)
            flag = False
            for window_idx in annotation_order:
                if self.max_budget is not None and self.count >= self.max_budget:
                    self.budget_reached = True
                    print(f"\n🛑 Early termination: Maximum pairwise constraint budget reached (max_budget={self.max_budget})")
                    print(
                        f"   Current count={self.count}, remaining {len(selected_nodes) - selected_nodes.index(node)} nodes unannotated")
                    print(f"   neighborhoods will no longer be updated, using current state for final clustering")
                    break

                self.count += 1

                if self.y[node] == self.y[self.neighborhoods[window_idx][0]]:
                    for view_idx in range(self.n_views):
                        self._add_node_to_window(view_idx, window_idx, node, centrality_maps[view_idx])

                    self.neighborhoods[window_idx].append(node)
                    flag = True
                    break
            if flag == False and not self.budget_reached:
                new_window_idx = len(self.neighborhoods)
                for view_idx in range(self.n_views):
                    self.windows[view_idx].append([node])
                    centrality_val = centrality_maps[view_idx].get(node, 0.0)
                    self.window_heaps[view_idx].append([(centrality_val, node)])
                self.neighborhoods.append([node])

    def _connection_cal(self, node, view_weights):
        closest_results = {}
        n_windows = len(self.windows[0]) if self.windows else 0
        for view_idx, X in enumerate(self.X_list):
            view_closest = {}
            node_feature = X[node].reshape(1, -1)
            for window_idx in range(n_windows):
                window = self.windows[view_idx][window_idx]
                window_node_ids = np.array(window, dtype=int)
                window_feats = X[window_node_ids].reshape(-1, node_feature.shape[1])
                distances = np.linalg.norm(window_feats - node_feature, axis=1)
                min_dist_idx = np.argmin(distances)
                min_distance = distances[min_dist_idx]
                view_closest[window_idx] = min_distance
            closest_results[view_idx] = view_closest
        normalized_distances = np.zeros((self.n_views, n_windows))
        for view_idx in range(self.n_views):
            view_dists = np.array([closest_results[view_idx][window_idx] for window_idx in range(n_windows)])
            d_min = view_dists.min()
            d_max = view_dists.max()
            if d_max - d_min < 1e-12:
                normalized_dists = np.zeros_like(view_dists)
            else:
                normalized_dists = (view_dists - d_min) / (d_max - d_min)
            normalized_distances[view_idx] = normalized_dists
        if view_weights is None:
            weights = np.ones(self.n_views) / self.n_views
        else:
            weights = view_weights
        fused_distances = np.sum(weights.reshape(-1, 1) * normalized_distances, axis=0)
        annotation_order = np.argsort(fused_distances).tolist()

        return annotation_order

    def _add_node_to_window(self, view_idx, window_idx, node, centrality_map):
        centrality_val = centrality_map.get(node, 0.0)
        heap = self.window_heaps[view_idx][window_idx]

        if len(heap) < self.window_size:
            heapq.heappush(heap, (centrality_val, node))
            self.windows[view_idx][window_idx].append(node)
        else:
            min_centrality, min_node = heap[0]
            if centrality_val > min_centrality:
                heapq.heapreplace(heap, (centrality_val, node))
                self.windows[view_idx][window_idx].remove(min_node)
                self.windows[view_idx][window_idx].append(node)
