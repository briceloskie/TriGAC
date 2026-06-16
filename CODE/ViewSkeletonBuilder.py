import networkx as nx
import numpy as np

class ViewSkeletonBuilder:


    def __init__(self):
        pass

    def clusters_division2(self, view_skeleton):
        undirected_G = view_skeleton.to_undirected()
        edges_with_weights = []
        for u, v, data in undirected_G.edges(data=True):
            weight = data.get('prob', 0.0)
            edges_with_weights.append((u, v, weight))
        if len(edges_with_weights) == 0:
            return [list(undirected_G.nodes())]
        edges_sorted = sorted(edges_with_weights, key=lambda x: x[2])
        weights = [e[2] for e in edges_sorted]
        if len(weights) == 1:
            return [list(undirected_G.nodes())]
        diffs = np.diff(weights)
        max_diff_idx = np.argmax(diffs)
        threshold = (weights[max_diff_idx] + weights[max_diff_idx + 1]) / 2
        if max_diff_idx < len(weights) - 1:
            relative_jump = diffs[max_diff_idx] / (weights[max_diff_idx] + 1e-8)
            if relative_jump < 0.2:
                return [list(undirected_G.nodes())]

        G_copy = undirected_G.copy()
        edges_to_remove = [(u, v) for u, v, w in edges_sorted if w < threshold]
        G_copy.remove_edges_from(edges_to_remove)

        clusters = [sorted(list(component)) for component in nx.connected_components(G_copy)]

        clusters.sort(key=len, reverse=True)

        return clusters

    import numpy as np
    import networkx as nx

    def clusters_division(self, view_skeleton):
        """
        Parameter-free view clustering based on Kneedle algorithm.
        Automatically determines splitting threshold by detecting "knee point" in weight distribution curve, without preset hyperparameters.
        """
        undirected_G = view_skeleton.to_undirected()

        # Extract edge weights (higher prob means more similar)
        edges_with_weights = []
        for u, v, data in undirected_G.edges(data=True):
            weight = data.get('prob', 0.0)
            edges_with_weights.append((u, v, weight))

        if len(edges_with_weights) == 0:
            return [list(undirected_G.nodes())]

        # Sort weights in ascending order
        edges_sorted = sorted(edges_with_weights, key=lambda x: x[2])
        weights = np.array([e[2] for e in edges_sorted])

        if len(weights) == 1:
            return [list(undirected_G.nodes())]

        # ===== Kneedle: Automatically detect natural breakpoint =====
        n = len(weights)

        # Normalize to [0, 1]
        x_norm = np.linspace(0, 1, n)
        w_min, w_max = weights.min(), weights.max()

        # If all weights are nearly equal, no clear cluster structure
        if w_max - w_min < 1e-8:
            return [list(undirected_G.nodes())]

        w_norm = (weights - w_min) / (w_max - w_min)

        # Construct reference line from first to last point
        line = w_norm[0] + (w_norm[-1] - w_norm[0]) * x_norm

        # Calculate perpendicular distance from curve to line (absolute value for compatibility with different curve shapes)
        distances = np.abs(line - w_norm)

        # Locate knee point (point with maximum deviation)
        knee_idx = int(np.argmax(distances))

        # If curve is approximately linear (no significant knee), keep as single cluster
        # Here 1e-2 is significance protection in normalized space
        if distances[knee_idx] < 1e-2:
            return [list(undirected_G.nodes())]

        # Ensure index does not go out of bounds
        if knee_idx >= n - 1:
            return [list(undirected_G.nodes())]

        # Use midpoint at knee as cutting threshold
        threshold = (weights[knee_idx] + weights[knee_idx + 1]) / 2

        # Remove edges below threshold (low similarity = inter-cluster connections)
        G_copy = undirected_G.copy()
        edges_to_remove = [(u, v) for u, v, w in edges_sorted if w < threshold]
        G_copy.remove_edges_from(edges_to_remove)

        # Return connected components as view clusters
        clusters = [sorted(list(c)) for c in nx.connected_components(G_copy)]
        clusters.sort(key=len, reverse=True)
        return clusters

    def build(self, view_similarity_matrix):
        view_skeleton = nx.DiGraph()
        view_roots = [i for i in range(len(view_similarity_matrix))]
        while True:
            sub_sim_matrix = view_similarity_matrix[np.array(view_roots)][:, np.array(view_roots)]
            sub_idx_to_view_root = {idx: root for idx, root in enumerate(view_roots)}
            closest_pairs = self._find_closest_views(sub_sim_matrix)
            sub_G = nx.DiGraph()
            for view_u_sub, view_v_sub, sim in closest_pairs:
                view_u_original = sub_idx_to_view_root[view_u_sub]
                view_v_original = sub_idx_to_view_root[view_v_sub]
                sub_G.add_edge(view_u_original, view_v_original, prob=sim)
            WCCs = [sub_G.subgraph(c).copy() for c in nx.weakly_connected_components(sub_G)]
            view_roots = []
            for WCC in WCCs:
                nnc = self._nnc_search(WCC)
                view_root = self._select_root3(WCC, nnc)
                view_roots.append(view_root)
                out_edges = list(WCC.out_edges(view_root))
                sub_G.remove_edges_from(out_edges)
            view_skeleton.add_edges_from(sub_G.edges(data=True))

            if len(view_roots) == 1:
                break
        return view_skeleton

    def _nnc_search(self,WCC):
        visited = set()  # Globally visited nodes (avoid duplicate processing)
        for node in WCC.nodes:
            if node in visited:
                continue
            path = []  # Record current traversal path (kept for final cycle return)
            node_to_path_idx = {}  # Dictionary: node -> its index in path (core optimization)
            current = node
            while True:
                # 1. If current node has been globally visited
                if current in visited:
                    # Check if it's in current path (dictionary lookup O(1))
                    if current in node_to_path_idx:
                        # Directly get index (dictionary lookup O(1)), no need to traverse list
                        idx = node_to_path_idx[current]
                        cycle = path[idx:]  # Extract cycle
                        return cycle
                    break
                # 2. Mark as visited and record in path and dictionary
                visited.add(current)
                node_to_path_idx[current] = len(path)  # Record current node's index
                path.append(current)
                # 3. Get next node (in your scenario, only 1 outgoing edge)
                neighbors = list(WCC.neighbors(current))
                if not neighbors:
                    break  # No neighbors, path terminates
                current = neighbors[0]
        return []  # Return empty list if no cycle

    def _select_root(self, component_graph, nnc):
        max_weight = -1.0
        root = None

        # Select node based on "sum of incoming edge similarity weights", if weights are equal select first encountered
        for node in nnc:
            in_edges = component_graph.in_edges(node, data=True)
            current_weight = sum([edge[2].get('prob', 0.0) for edge in in_edges])

            if current_weight > max_weight:
                max_weight = current_weight
                root = node

        # Extreme fallback: if NNC is empty, select node with smallest ID
        if root is None:
            nodes = sorted(component_graph.nodes())
            root = nodes[0] if nodes else None

        return root

    def _select_root3(self, component_graph, nnc):
        """
        Root selection strategy 3: Based on "shortest incoming edge from outside cycle"

        Core logic:
        1. Find all edges pointing from outside cycle to inside cycle
        2. Among these edges, find the one with maximum prob (similarity)
        3. The cycle node pointed to by this edge becomes the root

        Fallback: If no incoming edges from outside cycle, select node with smallest ID in NNC

        :param component_graph: NetworkX directed graph of current WCC
        :param nnc: Nearest Neighbor Cycle found in current WCC (node list)
        :return: root: selected root node
        """
        # 0. Boundary fallback: if NNC is empty, return node with smallest ID in graph
        if not nnc:
            return min(component_graph.nodes())

        # If NNC has only one node, return it directly
        if len(nnc) == 1:
            return nnc[0]

        # Convert NNC to set for O(1) lookup
        nnc_set = set(nnc)

        # 1. Collect all incoming edges from outside cycle to inside cycle
        incoming_edges = []
        for src, dst, data in component_graph.edges(data=True):
            # Condition: destination in cycle, source not in cycle
            if dst in nnc_set and src not in nnc_set:
                # Extract prob (similarity), handle possible missing key
                prob = data.get('prob', data.get('weight', 0.0))
                incoming_edges.append((prob, src, dst))

        # 2. Core logic: if there are incoming edges from outside cycle, select one with maximum prob
        if incoming_edges:
            # Sort by prob in descending order, take first one (highest similarity)
            incoming_edges.sort(key=lambda x: x[0], reverse=True)
            max_prob, max_src, max_dst = incoming_edges[0]
            return max_dst
        else:
            # 3. Fallback logic: if no incoming edges from outside cycle, select node with smallest ID in NNC
            return min(nnc)


    def _find_closest_views(self,sim_matrix):
        n_views = sim_matrix.shape[0]
        # 1. Set diagonal (similarity with itself) to -1, so argmax won't select itself
        sim_matrix_no_diag = sim_matrix.copy()
        np.fill_diagonal(sim_matrix_no_diag, -1)
        # 2. Take argmax for each row to find index of maximum value
        closest_indices = np.argmax(sim_matrix_no_diag, axis=1)
        # 3. Assemble results
        closest_pairs = []
        for i in range(n_views):
            j = closest_indices[i]
            sim = sim_matrix[i, j]
            closest_pairs.append((i, j, sim))
        return closest_pairs
