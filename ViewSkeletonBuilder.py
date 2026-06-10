import networkx as nx
import numpy as np

class ViewSkeletonBuilder:


    def __init__(self):
        pass

    def clusters_division2(self, view_skeleton):
        undirected_G = view_skeleton.to_undirected()

        # 步骤2：获取所有边的权重
        edges_with_weights = []
        for u, v, data in undirected_G.edges(data=True):
            weight = data.get('prob', 0.0)
            edges_with_weights.append((u, v, weight))

        # 如果没有边或只有一个节点，返回单簇
        if len(edges_with_weights) == 0:
            return [list(undirected_G.nodes())]

        # 步骤3：按权重从小到大排序
        edges_sorted = sorted(edges_with_weights, key=lambda x: x[2])
        weights = [e[2] for e in edges_sorted]

        # 步骤4：找到权重的"自然断点"
        # 方法：计算相邻权重的差值，找到最大的跳跃点
        if len(weights) == 1:
            # 只有一条边，不分割
            return [list(undirected_G.nodes())]

        diffs = np.diff(weights)

        # 找到差值最大的位置（权重跳跃最大的地方）
        max_diff_idx = np.argmax(diffs)
        threshold = (weights[max_diff_idx] + weights[max_diff_idx + 1]) / 2

        # 如果最大跳跃不够显著（相对变化<20%），则不分割
        if max_diff_idx < len(weights) - 1:
            relative_jump = diffs[max_diff_idx] / (weights[max_diff_idx] + 1e-8)
            if relative_jump < 0.2:  # 跳跃小于20%，认为没有明显的簇结构
                return [list(undirected_G.nodes())]

        # 步骤5：切断权重低于阈值的边
        G_copy = undirected_G.copy()
        edges_to_remove = [(u, v) for u, v, w in edges_sorted if w < threshold]
        G_copy.remove_edges_from(edges_to_remove)

        # 步骤6：获取连通分量作为簇
        clusters = [sorted(list(component)) for component in nx.connected_components(G_copy)]

        # 按簇的大小降序排列
        clusters.sort(key=len, reverse=True)

        return clusters

    import numpy as np
    import networkx as nx

    def clusters_division(self, view_skeleton):
        """
        基于 Kneedle 算法的无参视图分簇。
        通过检测权重分布曲线的“膝点”自动确定分割阈值，无需预设超参数。
        """
        undirected_G = view_skeleton.to_undirected()

        # 提取边权重（prob 越大越相似）
        edges_with_weights = []
        for u, v, data in undirected_G.edges(data=True):
            weight = data.get('prob', 0.0)
            edges_with_weights.append((u, v, weight))

        if len(edges_with_weights) == 0:
            return [list(undirected_G.nodes())]

        # 按权重升序排列
        edges_sorted = sorted(edges_with_weights, key=lambda x: x[2])
        weights = np.array([e[2] for e in edges_sorted])

        if len(weights) == 1:
            return [list(undirected_G.nodes())]

        # ===== Kneedle: 自动检测自然断点 =====
        n = len(weights)

        # 归一化到 [0,1]
        x_norm = np.linspace(0, 1, n)
        w_min, w_max = weights.min(), weights.max()

        # 如果所有权重几乎相等，说明没有明显的簇结构
        if w_max - w_min < 1e-8:
            return [list(undirected_G.nodes())]

        w_norm = (weights - w_min) / (w_max - w_min)

        # 构造首点尾点的参考直线
        line = w_norm[0] + (w_norm[-1] - w_norm[0]) * x_norm

        # 计算曲线到直线的垂直距离（取绝对值以兼容不同形状的曲线）
        distances = np.abs(line - w_norm)

        # 定位 knee point（最大偏离点）
        knee_idx = int(np.argmax(distances))

        # 若曲线近似线性（无显著 knee），则保持单簇
        # 这里的 1e-2 是归一化空间下的显著性保护
        if distances[knee_idx] < 1e-2:
            return [list(undirected_G.nodes())]

        # 确保索引不越界
        if knee_idx >= n - 1:
            return [list(undirected_G.nodes())]

        # 以 knee 处的中点作为切割阈值
        threshold = (weights[knee_idx] + weights[knee_idx + 1]) / 2

        # 切断低于阈值的边（低相似度 = 簇间连接）
        G_copy = undirected_G.copy()
        edges_to_remove = [(u, v) for u, v, w in edges_sorted if w < threshold]
        G_copy.remove_edges_from(edges_to_remove)

        # 返回连通分量作为视图簇
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
        return []  # 无环返回空列表


    def _select_root(self, component_graph, nnc):
        max_weight = -1.0
        root = None

        # 按「入边相似度权重加和」选点，权重相同选先遇到的
        for node in nnc:
            in_edges = component_graph.in_edges(node, data=True)
            current_weight = sum([edge[2].get('prob', 0.0) for edge in in_edges])

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
        选根策略3：基于"环外最短入边"选根

        核心逻辑：
        1. 找到所有从环外指向环内的边
        2. 在这些边中找到 prob（相似度）最大的那条
        3. 这条边指向的环内节点即为 root

        兜底：如果没有环外入边，则选 NNC 中 ID 最小的节点

        :param component_graph: 当前 WCC 的 NetworkX 有向图
        :param nnc: 当前 WCC 中找到的 Nearest Neighbor Cycle (节点列表)
        :return: root: 选中的根节点
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
                # 取出 prob（相似度），注意兼容可能的 key 缺失情况
                prob = data.get('prob', data.get('weight', 0.0))
                incoming_edges.append((prob, src, dst))

        # 2. 核心逻辑：如果有环外入边，选 prob 最大的
        if incoming_edges:
            # 按 prob 降序排序，取第一条（相似度最大的）
            incoming_edges.sort(key=lambda x: x[0], reverse=True)
            max_prob, max_src, max_dst = incoming_edges[0]
            return max_dst
        else:
            # 3. 兜底逻辑：如果没有环外入边，选 NNC 中 ID 最小的节点
            return min(nnc)


    def _find_closest_views(self,sim_matrix):
        n_views = sim_matrix.shape[0]
        # 1. 把对角线（自己和自己的相似度）设为 -1，这样argmax就不会选到自己了
        sim_matrix_no_diag = sim_matrix.copy()
        np.fill_diagonal(sim_matrix_no_diag, -1)
        # 2. 对每一行取 argmax，找到最大值的索引
        closest_indices = np.argmax(sim_matrix_no_diag, axis=1)
        # 3. 组装结果
        closest_pairs = []
        for i in range(n_views):
            j = closest_indices[i]
            sim = sim_matrix[i, j]
            closest_pairs.append((i, j, sim))
        return closest_pairs