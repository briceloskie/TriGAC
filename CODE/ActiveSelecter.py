from collections import defaultdict

import numpy as np


class ActiveSelecter:
    def __init__(self):
        self.unselected_nodes = []

    def node_selection_subsequent_round(self, budget, n_views, weights, uncertainties_dicts, centralities_dicts):
        budget = min(budget, len(self.unselected_nodes))
        if budget <= 0:
            print("Warning: Budget <= 0, no need to select nodes")
            return [], centralities_dicts

        if not uncertainties_dicts:
            print("Error: uncertainties_dicts is empty, cannot perform node selection!")
            return [], centralities_dicts
        if not centralities_dicts:
            print("Error: centralities_dicts is empty, cannot perform node selection!")
            return [], centralities_dicts

        view_candidates = {}

        for view_idx in range(n_views):
            cluster_uncertain_dict = uncertainties_dicts[view_idx]
            cluster_candidates = [(node, val, 'uncertainty') for node, val in cluster_uncertain_dict.items() if val > 0]

            need_supplement = max(0, budget - len(cluster_candidates))
            supplement_candidates = []

            if need_supplement > 0:
                conn_uncertain_dict = centralities_dicts[view_idx]
                cluster_nodes_set = set([n for n, _, _ in cluster_candidates])
                conn_candidates = [(node, val, 'centrality') for node, val in conn_uncertain_dict.items()
                                   if node not in cluster_nodes_set]
                supplement_candidates = conn_candidates[:need_supplement]

            final_candidates = cluster_candidates + supplement_candidates
            view_candidates[view_idx] = final_candidates[:budget]

        priority_pool_uncertainty = []
        priority_pool_centrality = []

        for view_idx in range(n_views):
            candidates = view_candidates[view_idx]
            if not candidates:
                continue

            uncertainty_items = [(node, score) for node, score, stype in candidates if stype == 'uncertainty']
            centrality_items = [(node, score) for node, score, stype in candidates if stype == 'centrality']

            if uncertainty_items:
                unc_scores = [score for _, score in uncertainty_items]
                max_unc_score = max(unc_scores) if unc_scores else 1.0
                if max_unc_score == 0:
                    max_unc_score = 1.0

                view_weight = weights[view_idx]
                for node, raw_score in uncertainty_items:
                    normalized_score = raw_score / max_unc_score
                    final_score = view_weight * normalized_score
                    priority_pool_uncertainty.append({
                        'node': node,
                        'view': view_idx,
                        'score': final_score,
                        'raw_score': raw_score,
                        'type': 'uncertainty'
                    })

            if centrality_items:
                cen_scores = [score for _, score in centrality_items]
                max_cen_score = max(cen_scores) if cen_scores else 1.0
                if max_cen_score == 0:
                    max_cen_score = 1.0

                view_weight = weights[view_idx]
                for node, raw_score in centrality_items:
                    normalized_score = raw_score / max_cen_score
                    final_score = view_weight * normalized_score
                    priority_pool_centrality.append({
                        'node': node,
                        'view': view_idx,
                        'score': final_score,
                        'raw_score': raw_score,
                        'type': 'centrality'
                    })

        selected_nodes = []
        selected_set = set()
        view_selection_count = {v: 0 for v in range(n_views)}

        priority_pool_uncertainty.sort(key=lambda x: x['score'], reverse=True)

        for item in priority_pool_uncertainty:
            if len(selected_nodes) >= budget:
                break
            node = item['node']
            view_idx = item['view']

            if node not in selected_set:
                selected_nodes.append(node)
                selected_set.add(node)
                view_selection_count[view_idx] += 1

        if len(selected_nodes) < budget:
            priority_pool_centrality.sort(key=lambda x: x['score'], reverse=True)

            for item in priority_pool_centrality:
                if len(selected_nodes) >= budget:
                    break
                node = item['node']
                view_idx = item['view']

                if node not in selected_set:
                    selected_nodes.append(node)
                    selected_set.add(node)
                    view_selection_count[view_idx] += 1

        if selected_nodes:
            for node in selected_nodes:
                for conn_dict in centralities_dicts:
                    conn_dict.pop(node, None)
                for clus_dict in uncertainties_dicts:
                    clus_dict.pop(node, None)

            unselected_set = set()
            for view_dict in centralities_dicts:
                unselected_set.update(view_dict.keys())
            self.unselected_nodes = sorted(list(unselected_set))

        return selected_nodes, centralities_dicts

    def node_selection_first_round(self, centralities, budget, n_views, n_samples):

        if not centralities or len(centralities) != n_views:
            print("Error: Please provide correct centrality list (length should equal number of views)!")
            return []

        budget = min(budget, n_samples)
        if budget <= 0:
            print("Warning: Budget <= 0, no need to select nodes")
            return []

        selected_nodes = []
        selected_set = set()

        rank_level = 0
        stop_flag = False

        while len(selected_nodes) < budget and not stop_flag:
            for view_idx in range(n_views):
                if len(selected_nodes) >= budget:
                    stop_flag = True
                    break

                current_sorted_dict = centralities[view_idx]
                sorted_nodes = list(current_sorted_dict.keys())

                if rank_level >= len(sorted_nodes):
                    continue

                candidate_node = sorted_nodes[rank_level]

                if candidate_node not in selected_set:
                    selected_nodes.append(candidate_node)
                    selected_set.add(candidate_node)

                    if len(selected_nodes) == budget:
                        stop_flag = True
                        break

            if not stop_flag:
                rank_level += 1

                max_nodes_in_any_view = max(len(d) for d in centralities) if centralities else 0
                if rank_level > max_nodes_in_any_view:
                    stop_flag = True

        if selected_nodes:

            before_delete_len = [len(d) for d in centralities]

            for node in selected_nodes:
                for view_dict in centralities:
                    view_dict.pop(node, None)

            after_delete_len = [len(d) for d in centralities]

            unselected_set = set()
            for view_dict in centralities:
                unselected_set.update(view_dict.keys())

            self.unselected_nodes = sorted(list(unselected_set))
        return selected_nodes, centralities

    def clustering_uncertainty_cal(self, local_regions, labeling_result, n_views, n_samples):

        uncertainties_dicts = []

        if not self.unselected_nodes:
            print(
                "Warning: Unselected nodes list (unselected_nodes) is empty, no need to calculate clustering uncertainty!")

        for view_idx in range(n_views):

            view_cluster_labels = labeling_result[view_idx]

            current_local_regions = local_regions[view_idx]

            view_uncertainty = {}

            for node in self.unselected_nodes:

                if node >= n_samples:
                    print(f"Warning: Node {node} exceeds total sample count {n_samples}, skipping")
                    continue

                knn_neighbors = current_local_regions[node]

                cluster_count = defaultdict(int)
                for neighbor_node in knn_neighbors:

                    neighbor_cluster = view_cluster_labels.get(neighbor_node, -1)

                    if neighbor_cluster == -1:
                        continue
                    cluster_count[neighbor_cluster] += 1

                entropy = 0.0

                if sum(cluster_count.values()) > 0:
                    for cluster_id in cluster_count:
                        total_valid = sum(cluster_count.values())
                        p = cluster_count[cluster_id] / total_valid
                        if p > 0:
                            entropy -= p * np.log2(p)

                view_uncertainty[node] = entropy

            def sort_key(item):
                node, uncertainty = item
                return -uncertainty

            sorted_items = sorted(view_uncertainty.items(), key=sort_key)
            sorted_node_dict = dict(sorted_items)
            uncertainties_dicts.append(sorted_node_dict)

        return uncertainties_dicts
