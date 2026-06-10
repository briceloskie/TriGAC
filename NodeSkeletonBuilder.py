import heapq

import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree


# ===================== 你的NodeSkeletonBuilder（保持不变，已经生成树） =====================
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
                    node_root = self._select_root3(current_wcc, nnc) # 方法 3
                    out_edges = list(current_wcc.out_edges(node_root))
                    current_wcc.remove_edges_from(out_edges)
                    node_skeleton.add_edges_from(current_wcc.edges(data=True))
                    new_node_roots.append(node_root)
                node_roots = new_node_roots
            # 计算每个点的代表性。
            # node_skeleton=self._centrality_cal_way1(node_skeleton, node_roots[0])
            # node_skeleton=self._centrality_cal_way2(node_skeleton, node_roots[0])
            node_skeleton = self._centrality_cal_way3(node_skeleton, node_roots[0])
            # node_skeleton = self._centrality_cal_way2(node_skeleton, node_roots[0])
            # 提取每个点的centrality，存入字典并排序
            centrality_dict = self._extract_and_sort_centrality(node_skeleton)
            centralities_dicts.append(centrality_dict)
            node_skeletons.append(node_skeleton)

        return node_skeletons,centralities_dicts

    def _extract_and_sort_centrality(self, skeleton):
        """
        提取图中每个节点的centrality值，存入字典并按值从大到小排序

        参数:
            skeleton: networkx图对象，节点包含'centrality'属性

        返回:
            sorted_centrality_dict: 按centrality值从大到小排序的字典 {node_id: centrality_value}
        """
        # 提取所有节点的centrality值到字典
        centrality_dict = {}
        for node in skeleton.nodes():
            if 'centrality' in skeleton.nodes[node]:
                centrality_dict[node] = skeleton.nodes[node]['centrality']

        # 按centrality值从大到小排序
        sorted_centrality_dict = dict(sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_centrality_dict

    # def _centrality_ranking(self, G, start_node):
    #     traversed_nodes = [start_node]
    #     visited = {start_node}
    #     candidate_edge = []
    #
    #     # 将起始节点的出边加入候选队列
    #     for edge in G.out_edges(start_node, data=True):
    #         heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    #
    #     while candidate_edge:
    #         max_edge = heapq.heappop(candidate_edge)
    #         weight, current_node, new_node = -max_edge[0], max_edge[1], max_edge[2]
    #
    #         # 如果新节点未被访问过，则加入遍历序列
    #         if new_node not in visited:
    #             visited.add(new_node)
    #             traversed_nodes.append(new_node)
    #
    #             # 将新节点的出边加入候选队列
    #             for edge in G.out_edges(new_node, data=True):
    #                 if edge[1] not in visited:
    #                     heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    #
    #     return traversed_nodes

    def _centrality_cal_way3(self, skeleton, start_node):
        """
        基于出边长度的中心性计算

        中心性定义为节点所有出边的距离之和：
        centrality(x_i) = Σ_{x_k ∈ N_out(x_i)} distance(x_i, x_k)

        距离越大表示该节点连接到的邻居越远，代表性越强

        参数:
            skeleton: networkx有向图对象
            start_node: 根节点（给予最高优先级）

        返回:
            skeleton: 添加了centrality属性的图对象
        """
        for node in skeleton.nodes():
            if node == start_node:
                # 根节点：使用一个很大的固定值
                skeleton.nodes[node]['centrality'] = 1e6
            else:
                # 计算出边的距离之和：Σ distance(x_i, x_k)
                out_edges = list(skeleton.out_edges(node, data=True))
                out_distance_sum = sum([
                    edge[2].get('distance', 0.0) for edge in out_edges
                ])

                # 中心性 = 出边长度之和
                skeleton.nodes[node]['centrality'] = out_distance_sum

        return skeleton

    def _centrality_cal_way2(self, skeleton, start_node):
        """
        基于加权度数的中心性计算
        - 根节点赋予最高优先级
        - 其他节点使用加权入度 + 出度距离
        """
        for node in skeleton.nodes():
            if node == start_node:
                # 根节点：使用一个很大的固定值（而非inf）
                skeleton.nodes[node]['centrality'] = 1e6
            else:
                # 入度：考虑入边的相似度权重
                in_edges = list(skeleton.in_edges(node, data=True))
                weighted_in_degree = sum([
                    edge[2].get('similarity', 0.5) for edge in in_edges
                ])

                # 出度：考虑出边的距离（距离越大，代表性越强）
                out_edges = list(skeleton.out_edges(node, data=True))
                out_distance_sum = sum([
                    edge[2].get('distance', 0.0) for edge in out_edges
                ])
                # 综合评分
                skeleton.nodes[node]['centrality'] = weighted_in_degree + out_distance_sum
        return skeleton

    def _centrality_cal_way1(self, skeleton, start_node):
        visited_nodes = set()
        traversal_order = []
        candidate_edges = []

        # 1. 初始化起始节点及候选边
        visited_nodes.add(start_node)
        traversal_order.append(start_node)

        # 加载起始节点的入边 (v -> u, 其中 u 是 start_node, v 是子节点)
        for v, u, data in skeleton.in_edges(start_node, data=True):
            dist = data.get('distance', data.get('weight', 0.0))
            # 取负值存入小顶堆，实现大值优先
            heapq.heappush(candidate_edges, (-dist, u, v))

        # 2. 贪心遍历：按 distance 从大到小扩展
        while candidate_edges:
            neg_dist, current_parent, next_child = heapq.heappop(candidate_edges)

            if next_child in visited_nodes:
                continue

            visited_nodes.add(next_child)
            traversal_order.append(next_child)

            # 加载新访问节点的入边
            for v_child, v_parent, data_new in skeleton.in_edges(next_child, data=True):
                if v_child not in visited_nodes:
                    dist_new = data_new.get('distance', data_new.get('weight', 0.0))
                    heapq.heappush(candidate_edges, (-dist_new, v_parent, v_child))

        # 3. 处理孤立节点（如果有的话）
        isolated_nodes = [node for node in skeleton.nodes() if node not in visited_nodes]
        traversal_order.extend(sorted(isolated_nodes))
        # 4. 赋值 centrality/ranking
        for i, node in enumerate(traversal_order):
            skeleton.nodes[node]['centrality'] = 1 / (i + 1)
            skeleton.nodes[node]['ranking'] = i
        return skeleton

    def _nearest_neighbor_search(self, features_list):
        if not isinstance(features_list, np.ndarray):
            raise ValueError("features_list 必须是 numpy 数组")
        if len(features_list.shape) != 2:
            raise ValueError("features_list 必须是二维数组，形状为 (n_samples, n_features)")
        n_samples = len(features_list)
        if n_samples < 2:
            raise ValueError("features_list 至少需要包含 2 个点才能计算最近邻")
        # ========== 替换为BallTree实现（核心修改） ==========
        # 构建BallTree，使用L2距离（和原hnswlib的space='l2'一致）
        ball_tree = BallTree(features_list, metric='l2')
        # 查询每个点的k=2个最近邻（距离+索引）
        # BallTree.query返回：(距离数组, 索引数组)，和hnswlib的(labels, distances)顺序相反
        distances, indices = ball_tree.query(features_list, k=2)
        nn_results = []
        for idx in range(n_samples):
            candidate_indices = indices[idx]  # 对应原hnswlib的labels[idx]
            candidate_distances = distances[idx]  # 对应原hnswlib的distances[idx]
            # 逻辑和原代码完全一致：排除自身，取另一个最近邻
            if candidate_indices[0] == idx:
                neighbor_idx = candidate_indices[1]
                neighbor_distance = candidate_distances[1]
            else:
                neighbor_idx = candidate_indices[0]
                neighbor_distance = candidate_distances[0]
            nn_results.append([idx, neighbor_idx, neighbor_distance])
        return nn_results

    def _nnc_search(self,WCC):
        visited = set()  # 全局已访问节点（避免重复处理）
        for node in WCC.nodes:
            if node in visited:
                continue
            path = []  # 记录当前遍历路径（保留，用于最终返回环）
            node_to_path_idx = {}  # 字典：节点 -> 其在path中的索引（核心优化）
            current = node
            while True:
                # 1. 若当前节点已全局访问过
                if current in visited:
                    # 检查是否在当前路径中（字典查询 O(1)）
                    if current in node_to_path_idx:
                        # 直接取索引（字典查询 O(1)），无需遍历列表
                        idx = node_to_path_idx[current]
                        cycle = path[idx:]  # 提取环
                        return cycle
                    break
                # 2. 标记为已访问，并记录到路径和字典中
                visited.add(current)
                node_to_path_idx[current] = len(path)  # 记录当前节点的索引
                path.append(current)
                # 3. 获取下一个节点（你的场景中仅1条出边）
                neighbors = list(WCC.neighbors(current))
                if not neighbors:
                    break  # 无邻居，路径终止
                current = neighbors[0]
        return []

    def _select_root(self,component_graph, nnc):
        max_representativeness = -1
        root = None
        for node in nnc:
            # 关键修正：用有向图的入度（in_degree）替代总度数
            in_degree = component_graph.in_degree(node)
            representativeness = in_degree
            # 优化1：平局时，优先选nnc中靠前的节点（利用nnc的原有排序）
            if representativeness > max_representativeness or (
                    representativeness == max_representativeness and root is None):
                max_representativeness = representativeness
                root = node
        return root

    def _select_root2(self, component_graph, nnc):
        max_weight = -1.0
        root = None
        # 仅按「入边相似度权重加和」选点，权重相同选先遇到的
        for node in nnc:
            in_edges = component_graph.in_edges(node, data=True)
            current_weight = sum([edge[2].get('similarity', 0.0) for edge in in_edges])
            if current_weight > max_weight:
                max_weight = current_weight
                root = node
        # 极端兜底：NNC为空时选ID最小的节点
        if root is None:
            nodes = sorted(component_graph.nodes())
            root = nodes[0] if nodes else None
        return root

    def _select_root3(self, component_graph, nnc):
        """
        选根策略3：基于“环外最短入边”选根
        逻辑：
        1. 找到所有从环外指向环内的边
        2. 在这些边中找到 distance 最小的那条
        3. 这条边指向的环内节点即为 root
        兜底：如果没有环外入边，则选 NNC 中 ID 最小的节点

        参数:
            component_graph: 当前 WCC 的 NetworkX 有向图
            nnc: 当前 WCC 中找到的 Nearest Neighbor Cycle (节点列表)

        返回:
            root: 选中的根节点
        """
        # 0. 边界兜底：如果 NNC 为空，返回全图 ID 最小的节点
        if not nnc:
            return min(component_graph.nodes())
        # 如果 NNC 只有一个点，直接返回
        if len(nnc) == 1:
            return nnc[0]

        # 把 NNC 转成集合，方便后续 O(1) 查找
        nnc_set = set(nnc)

        # 1. 收集所有「从环外指向环内」的邻接边
        incoming_edges = []
        for src, dst, data in component_graph.edges(data=True):
            # 条件：终点在环内，且起点不在环内
            if dst in nnc_set and src not in nnc_set:
                # 取出 distance，注意兼容可能的 key 缺失情况
                dist = data.get('distance', data.get('weight', float('inf')))
                incoming_edges.append((dist, src, dst))

        # 2. 核心逻辑：如果有环外入边，选 distance 最小的
        if incoming_edges:
            # 按 distance 排序，取第一条
            incoming_edges.sort(key=lambda x: x[0])
            min_dist, min_src, min_dst = incoming_edges[0]
            return min_dst
        else:
            # 3. 兜底逻辑：如果没有环外入边，选 NNC 中 ID 最小的节点
            return min(nnc)




