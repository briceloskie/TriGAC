from collections import defaultdict
import networkx as nx


class NodeSkeletonRefiner:
    def __init__(self):
        self.labeling_result = []
        self.view_neighborhood_groups = []
        self.node_final_neigh = {}

    def refine(self, node_skeletons, neighborhoods, view_weights):
        target_nodes = set()
        for neighborhood in neighborhoods:
            target_nodes.update(neighborhood)

        for idx, skeleton in enumerate(node_skeletons):
            for node in target_nodes:
                if node not in skeleton.nodes:
                    continue
                out_edges = list(skeleton.out_edges(node))
                if out_edges:
                    skeleton.remove_edges_from(out_edges)

        self.labeling_result = []
        self.view_neighborhood_groups = []

        for skeleton_idx, skeleton in enumerate(node_skeletons):
            node2label = {node: -1 for node in skeleton.nodes()}
            current_view_groups = defaultdict(list)
            component_cache = {}
            if isinstance(skeleton, nx.DiGraph):
                components = list(nx.weakly_connected_components(skeleton))
            else:
                components = list(nx.connected_components(skeleton))

            for comp in components:
                for node in comp:
                    component_cache[node] = comp

            for neighborhood_idx, neighborhood in enumerate(neighborhoods):
                for node in neighborhood:
                    if node not in component_cache:
                        continue
                    connected_nodes = component_cache[node]

                    for conn_node in connected_nodes:
                        node2label[conn_node] = neighborhood_idx

                        current_view_groups[neighborhood_idx].append(conn_node)

            for neigh_id in current_view_groups:
                current_view_groups[neigh_id] = list(set(current_view_groups[neigh_id]))
            self.view_neighborhood_groups.append(dict(current_view_groups))

            self.labeling_result.append(node2label)

        if not self.labeling_result:
            print("⚠️ Warning: labeling_result is empty, please check if node_skeletons are valid!")
            return {}, {}, []

        node_neigh_weight = self.calculate_weight_sum(view_weights)

        return node_skeletons

    def calculate_weight_sum(self, view_weights):

        n_views = len(self.labeling_result)
        if view_weights.shape[0] != n_views:
            raise ValueError(
                f"Number of views in weight vector ({view_weights.shape[0]}) does not match number of views in labeling_result ({n_views})!")

        n_samples = len(self.labeling_result[0])
        node_neigh_weight = defaultdict(lambda: defaultdict(float))
        for view_idx in range(n_views):
            view_node2label = self.labeling_result[view_idx]
            view_weight = view_weights[view_idx]
            for node_id in range(n_samples):

                neigh_label = view_node2label.get(node_id, -1)
                if neigh_label == -1:
                    continue
                node_neigh_weight[node_id][neigh_label] += view_weight

        self.node_final_neigh = {}
        for node_id, neigh_weight_dict in node_neigh_weight.items():
            if not neigh_weight_dict:
                self.node_final_neigh[node_id] = -1
                continue
            final_neigh = max(neigh_weight_dict.items(), key=lambda x: x[1])[0]
            self.node_final_neigh[node_id] = final_neigh

        return dict(node_neigh_weight)
