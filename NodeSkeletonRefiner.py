from collections import defaultdict
import networkx as nx




class NodeSkeletonRefiner:
    def __init__(self):
        self.labeling_result = []
        self.view_neighborhood_groups=[]
        self.node_final_neigh= {}


    def refine(self, node_skeletons, neighborhoods, view_weights):
        target_nodes = set()
        for neighborhood in neighborhoods:
            target_nodes.update(neighborhood)

        for idx, skeleton in enumerate(node_skeletons):
            # 遍历所有目标节点
            for node in target_nodes:
                if node not in skeleton.nodes:
                    continue  # 节点不在当前图中则跳过
                out_edges = list(skeleton.out_edges(node))
                if out_edges:
                    skeleton.remove_edges_from(out_edges)

        self.labeling_result = []  # 存储每个视图的node2label
        self.view_neighborhood_groups = []  # 【新增】每个视图的分组：[视图0: {邻域ID: [节点列表]}, 视图1: {...}, ...]

        for skeleton_idx, skeleton in enumerate(node_skeletons):
            node2label = {node: -1 for node in skeleton.nodes()}
            # 【新增】初始化当前视图的邻域分组（邻域ID→节点列表）
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
                    # 分配标签：连通分量内所有节点归为当前neighborhood_idx
                    for conn_node in connected_nodes:
                        node2label[conn_node] = neighborhood_idx
                        # 【新增】把节点加入当前视图的对应邻域分组
                        current_view_groups[neighborhood_idx].append(conn_node)

            for neigh_id in current_view_groups:
                current_view_groups[neigh_id] = list(set(current_view_groups[neigh_id]))
            self.view_neighborhood_groups.append(dict(current_view_groups))


            self.labeling_result.append(node2label)

        # 防御性检查：避免空列表导致后续报错
        if not self.labeling_result:
            print("⚠️ 警告：labeling_result为空，请检查node_skeletons是否有效！")
            return {}, {}, []

        # 计算权重加和（原有逻辑）
        node_neigh_weight= self.calculate_weight_sum(view_weights)

        # 返回值新增：每个视图的邻域分组（融合前的独立分组结果）
        return node_skeletons
    def calculate_weight_sum(self, view_weights):
        """
        核心权重加和函数（视图级权重版本）
        :param labeling_result: 原有视图级节点-邻域标签映射
        :param sample_scale_weights: 视图级权重向量 (n_views,)
        :return:
            node_neigh_weight: dict {节点ID: {邻域ID: 总权重}}
            node_final_neigh: dict {节点ID: 最终归属邻域ID}
        """
        # 防御性检查：视图数匹配
        n_views = len(self.labeling_result)
        if view_weights.shape[0] != n_views:
            raise ValueError(
                f"权重向量的视图数({view_weights.shape[0]})与labeling_result的视图数({n_views})不匹配！")

        # 获取所有节点数量（从任意一个视图的labeling_result中获取）
        n_samples = len(self.labeling_result[0])

        # 初始化：节点→{邻域: 权重和}
        node_neigh_weight = defaultdict(lambda: defaultdict(float))

        # 遍历每个视图，累加权重
        for view_idx in range(n_views):
            # 当前视图的节点-邻域映射
            view_node2label = self.labeling_result[view_idx]
            # 获取当前视图的全局权重
            view_weight = view_weights[view_idx]
            # 遍历所有节点
            for node_id in range(n_samples):
                # 跳过无邻域标签的节点（label=-1）
                neigh_label = view_node2label.get(node_id, -1)
                if neigh_label == -1:
                    continue
                # 累加：节点+邻域 维度的权重（使用全局视图权重）
                node_neigh_weight[node_id][neigh_label] += view_weight

        # 基于权重和，确定每个节点最终归属的邻域（取权重最大的邻域）
        self.node_final_neigh = {}
        for node_id, neigh_weight_dict in node_neigh_weight.items():
            if not neigh_weight_dict:  # 无任何邻域权重的节点，归为-1
                self.node_final_neigh[node_id] = -1
                continue
            # 取权重最大的邻域标签
            final_neigh = max(neigh_weight_dict.items(), key=lambda x: x[1])[0]
            self.node_final_neigh[node_id] = final_neigh

        return dict(node_neigh_weight)

