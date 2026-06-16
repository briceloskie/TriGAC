import heapq

import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree


class NodeSkeletonBuilder:
    def __init__(self):
        pass

    def build(self, X_list):
        node_skeletons=[]
        centralities_dicts=[]
        for X in X_list:
            node_skeleton = nx.DiGraph()
            initial_nodes = [i for i in range(len(X_list[0]))]
            node_skeleton.add_nodes_from(initial_nodes)
            node_roots = initial_nodes.copy()
            while True:
                if len(node_roots) == 1:
                    break
                node_roots_to_ids_mapping = {list_idx: node_id for list_idx, node_id in enumerate(node_roots)}
                X_subset = X[node_roots]
                nn_results = self._nearest_neighbor_search(X_subset)
                sub_wcc = nx.DiGraph()
                for edge in nn_results:
                    src = node_roots_to_ids_mapping[edge[0]]
                    dst = node_roots_to_ids_mapping[edge[1]]
                    distance = edge[2]
                    similarity = 1 / (1+distance + 1e-12)
                    sub_wcc.add_edge(src, dst, distance=distance, similarity=similarity,weight=distance)
                wccs = [sub_wcc.subgraph(c).copy() for c in nx.weakly_connected_components(sub_wcc)]
                new_node_roots = []
                for wcc in wccs:
                    current_wcc = wcc.copy()
                    nnc = self._nnc_search(current_wcc)
                    node_root = self._select_root3(current_wcc, nnc) # Method 3
                    out_edges = list(current_wcc.out_edges(node_root))
                    current_wcc.remove_edges_from(out_edges)
                    node_skeleton.add_edges_from(current_wcc.edges(data=True))
                    new_node_roots.append(node_root)
                node_roots = new_node_roots
            node_skeleton = self._centrality_cal_way3(node_skeleton, node_roots[0])
            centrality_dict = self._extract_and_sort_centrality(node_skeleton)
            centralities_dicts.append(centrality_dict)
            node_skeletons.append(node_skeleton)

        return node_skeletons,centralities_dicts

    def _extract_and_sort_centrality(self, skeleton):
        centrality_dict = {}
        for node in skeleton.nodes():
            if 'centrality' in skeleton.nodes[node]:
                centrality_dict[node] = skeleton.nodes[node]['centrality']

        sorted_centrality_dict = dict(sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_centrality_dict


    def _centrality_cal_way3(self, skeleton, start_node):
        for node in skeleton.nodes():
            if node == start_node:
                skeleton.nodes[node]['centrality'] = 1e6
            else:
                out_edges = list(skeleton.out_edges(node, data=True))
                out_distance_sum = sum([
                    edge[2].get('distance', 0.0) for edge in out_edges
                ])

                # Centrality = sum of outgoing edge lengths
                skeleton.nodes[node]['centrality'] = out_distance_sum

        return skeleton

    def _centrality_cal_way2(self, skeleton, start_node):
        for node in skeleton.nodes():
            if node == start_node:
                skeleton.nodes[node]['centrality'] = 1e6
            else:
                in_edges = list(skeleton.in_edges(node, data=True))
                weighted_in_degree = sum([
                    edge[2].get('similarity', 0.5) for edge in in_edges
                ])

                out_edges = list(skeleton.out_edges(node, data=True))
                out_distance_sum = sum([
                    edge[2].get('distance', 0.0) for edge in out_edges
                ])
                skeleton.nodes[node]['centrality'] = weighted_in_degree + out_distance_sum
        return skeleton

    def _centrality_cal_way1(self, skeleton, start_node):
        visited_nodes = set()
        traversal_order = []
        candidate_edges = []
        visited_nodes.add(start_node)
        traversal_order.append(start_node)

        for v, u, data in skeleton.in_edges(start_node, data=True):
            dist = data.get('distance', data.get('weight', 0.0))
            heapq.heappush(candidate_edges, (-dist, u, v))

        while candidate_edges:
            neg_dist, current_parent, next_child = heapq.heappop(candidate_edges)

            if next_child in visited_nodes:
                continue

            visited_nodes.add(next_child)
            traversal_order.append(next_child)

            for v_child, v_parent, data_new in skeleton.in_edges(next_child, data=True):
                if v_child not in visited_nodes:
                    dist_new = data_new.get('distance', data_new.get('weight', 0.0))
                    heapq.heappush(candidate_edges, (-dist_new, v_parent, v_child))

        isolated_nodes = [node for node in skeleton.nodes() if node not in visited_nodes]
        traversal_order.extend(sorted(isolated_nodes))
        for i, node in enumerate(traversal_order):
            skeleton.nodes[node]['centrality'] = 1 / (i + 1)
            skeleton.nodes[node]['ranking'] = i
        return skeleton

    def _nearest_neighbor_search(self, features_list):
        if not isinstance(features_list, np.ndarray):
            raise ValueError("features_list must be a numpy array")
        if len(features_list.shape) != 2:
            raise ValueError("features_list must be a 2D array with shape (n_samples, n_features)")
        n_samples = len(features_list)
        if n_samples < 2:
            raise ValueError("features_list must contain at least 2 points to compute nearest neighbor")
        ball_tree = BallTree(features_list, metric='l2')
        distances, indices = ball_tree.query(features_list, k=2)
        nn_results = []
        for idx in range(n_samples):
            candidate_indices = indices[idx]
            candidate_distances = distances[idx]
            if candidate_indices[0] == idx:
                neighbor_idx = candidate_indices[1]
                neighbor_distance = candidate_distances[1]
            else:
                neighbor_idx = candidate_indices[0]
                neighbor_distance = candidate_distances[0]
            nn_results.append([idx, neighbor_idx, neighbor_distance])
        return nn_results

    def _nnc_search(self,WCC):
        visited = set()
        for node in WCC.nodes:
            if node in visited:
                continue
            path = []
            node_to_path_idx = {}
            current = node
            while True:
                if current in visited:
                    if current in node_to_path_idx:
                        idx = node_to_path_idx[current]
                        cycle = path[idx:]
                        return cycle
                    break
                visited.add(current)
                node_to_path_idx[current] = len(path)
                path.append(current)
                neighbors = list(WCC.neighbors(current))
                if not neighbors:
                    break
                current = neighbors[0]
        return []

    def _select_root(self,component_graph, nnc):
        max_representativeness = -1
        root = None
        for node in nnc:
            in_degree = component_graph.in_degree(node)
            representativeness = in_degree
            if representativeness > max_representativeness or (
                    representativeness == max_representativeness and root is None):
                max_representativeness = representativeness
                root = node
        return root

    def _select_root2(self, component_graph, nnc):
        max_weight = -1.0
        root = None
        for node in nnc:
            in_edges = component_graph.in_edges(node, data=True)
            current_weight = sum([edge[2].get('similarity', 0.0) for edge in in_edges])
            if current_weight > max_weight:
                max_weight = current_weight
                root = node
        if root is None:
            nodes = sorted(component_graph.nodes())
            root = nodes[0] if nodes else None
        return root

    def _select_root3(self, component_graph, nnc):
        if not nnc:
            return min(component_graph.nodes())
        if len(nnc) == 1:
            return nnc[0]

        nnc_set = set(nnc)

        incoming_edges = []
        for src, dst, data in component_graph.edges(data=True):

            if dst in nnc_set and src not in nnc_set:

                dist = data.get('distance', data.get('weight', float('inf')))
                incoming_edges.append((dist, src, dst))


        if incoming_edges:
            incoming_edges.sort(key=lambda x: x[0])
            min_dist, min_src, min_dst = incoming_edges[0]
            return min_dst
        else:
            return min(nnc)
