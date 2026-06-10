from collections import defaultdict

import numpy as np  # 新增：导入numpy以使用inf

class ActiveSelecter:
    def __init__(self):
        # self.sorted_clustering_uncertainty_dicts_per_view = []
        self.unselected_nodes=[]

    def node_selection_subsequent_round(self, budget, n_views, weights, uncertainties_dicts, centralities_dicts):
        # ========== 防御性检查 ==========
        budget = min(budget, len(self.unselected_nodes))
        if budget <= 0:
            print("警告：预算≤0，无需选择节点")
            return [], centralities_dicts

        # 检查聚类/连接不确定性是否初始化
        if not uncertainties_dicts:
            print("错误：uncertainties_dicts为空，无法进行节点选择！")
            return [], centralities_dicts
        if not centralities_dicts:
            print("错误：centralities_dicts为空，无法进行节点选择！")
            return [], centralities_dicts

        # ========== 步骤 1：视图内初选（每个视图选 budget 个候选） ==========
        # 结构：view_candidates[view_idx] = [(node, raw_score, score_type), ...]
        # score_type: 'uncertainty' 或 'centrality'，用于标记节点来源
        view_candidates = {}

        for view_idx in range(n_views):
            # 1.1 提取 clustering_uncertainty > 0 的节点（已按降序）
            cluster_uncertain_dict = uncertainties_dicts[view_idx]
            cluster_candidates = [(node, val, 'uncertainty') for node, val in cluster_uncertain_dict.items() if val > 0]

            # 1.2 如果不够 budget，用 connection_uncertainty (representativeness) 补充
            need_supplement = max(0, budget - len(cluster_candidates))
            supplement_candidates = []

            if need_supplement > 0:
                conn_uncertain_dict = centralities_dicts[view_idx]
                # 排除已经在聚类候选中的节点
                cluster_nodes_set = set([n for n, _, _ in cluster_candidates])
                conn_candidates = [(node, val, 'centrality') for node, val in conn_uncertain_dict.items()
                                   if node not in cluster_nodes_set]
                # 按 connection_uncertainty 降序取前 need_supplement 个
                supplement_candidates = conn_candidates[:need_supplement]

            # 1.3 合并：聚类候选（不确定性） + 补充候选（中心性）
            final_candidates = cluster_candidates + supplement_candidates
            view_candidates[view_idx] = final_candidates[:budget]  # 确保不超过 budget

        # ========== 步骤 2：全局统一评分（分优先级处理） ==========
        # 关键改进：将候选池分为两个优先级层级
        priority_pool_uncertainty = []  # 优先级1：不确定性节点
        priority_pool_centrality = []  # 优先级2：中心性补充节点

        for view_idx in range(n_views):
            candidates = view_candidates[view_idx]
            if not candidates:
                continue

            # 分离两种类型的节点
            uncertainty_items = [(node, score) for node, score, stype in candidates if stype == 'uncertainty']
            centrality_items = [(node, score) for node, score, stype in candidates if stype == 'centrality']

            # 对不确定性节点归一化并加权
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

            # 对中心性节点归一化并加权
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

        # ========== 步骤 3：按优先级全局排序并选择 ==========
        # 关键：先选所有不确定性节点，再选中心性节点
        selected_nodes = []
        selected_set = set()
        view_selection_count = {v: 0 for v in range(n_views)}

        # 优先级1：先按得分排序不确定性节点
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

        # 优先级2：如果还没凑够，再从中心性节点中补充
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

        # ========== 步骤 4：更新未选中节点 ==========
        if selected_nodes:
            # 删除连接不确定性字典中的选中节点
            for node in selected_nodes:
                for conn_dict in centralities_dicts:
                    conn_dict.pop(node, None)
                # 聚类字典也清理一下
                for clus_dict in uncertainties_dicts:
                    clus_dict.pop(node, None)

            # 重新收集未选中节点
            unselected_set = set()
            for view_dict in centralities_dicts:
                unselected_set.update(view_dict.keys())
            self.unselected_nodes = sorted(list(unselected_set))

        return selected_nodes, centralities_dicts

    def node_selection_first_round(self, centralities, budget, n_views, n_samples):
        """
        【核心逻辑：基于centrality从大到小选点，按排名层级遍历多视图，凑够budget立即停止】
        流程：
            1. 按排名层级遍历所有视角，每找到1个未选中节点就加入
            2. 只要选中节点数达到budget，立即停止所有遍历（不管当前层级是否遍历完所有视角）
            3. 最后统一删除选中的budget个节点
        """
        # 防御性检查：确保centrality字典已提供
        if not centralities or len(centralities) != n_views:
            print("错误：请提供正确的centrality列表（长度应等于视图数）！")
            return []

        # 预算修正：不能超过总节点数
        budget = min(budget, n_samples)
        if budget <= 0:
            print("警告：预算≤0，无需选择节点")
            return []

        selected_nodes = []  # 最终选中的节点列表（按选择顺序）
        selected_set = set()  # 查重集合（O(1) 查重）

        # ========== 第一步：只读遍历，凑够budget立即停止 ==========
        rank_level = 0  # 排名层级（从0开始，0=最高优先级）
        stop_flag = False  # 停止遍历的标记

        while len(selected_nodes) < budget and not stop_flag:
            # 遍历所有视角（但可能中途停止）
            for view_idx in range(n_views):
                # 一旦凑够budget，立即停止当前视角遍历
                if len(selected_nodes) >= budget:
                    stop_flag = True
                    break

                # 取出当前视角的centrality排序字典（只读，不修改）
                current_sorted_dict = centralities[view_idx]
                # 转列表获取排序后的节点（按centrality降序）
                sorted_nodes = list(current_sorted_dict.keys())

                # 跳过：当前排名层级超出该视角的节点数
                if rank_level >= len(sorted_nodes):
                    continue

                # 获取当前排名的节点
                candidate_node = sorted_nodes[rank_level]

                # 查重：未选中则加入
                if candidate_node not in selected_set:
                    selected_nodes.append(candidate_node)
                    selected_set.add(candidate_node)

                    # 关键：只要凑够budget，立即标记停止
                    if len(selected_nodes) == budget:
                        stop_flag = True
                        break

            # 若未凑够，进入下一个排名层级（优先级降低）
            if not stop_flag:
                rank_level += 1
                # 防御：所有层级遍历完仍未凑够（极端情况）
                max_nodes_in_any_view = max(len(d) for d in centralities) if centralities else 0
                if rank_level > max_nodes_in_any_view:
                    stop_flag = True
                    # print(f"警告：遍历完所有排名层级仅选中{len(selected_nodes)}个节点，未达到预算{budget}")

        # ========== 第二步：统一删除选中的节点（仅删除最终选中的节点） ==========
        if selected_nodes:
            # 记录删除前的节点数（方便验证）
            before_delete_len = [len(d) for d in centralities]

            # 从centrality字典中删除选中的节点
            for node in selected_nodes:
                for view_dict in centralities:
                    view_dict.pop(node, None)  # pop 不存在的节点不报错

            # 记录删除后的节点数
            after_delete_len = [len(d) for d in centralities]

            # 收集所有视角中剩余的未选中节点（去重）
            unselected_set = set()
            for view_dict in centralities:
                unselected_set.update(view_dict.keys())
            # 转换为列表并排序（保持一致性）
            self.unselected_nodes = sorted(list(unselected_set))
        return selected_nodes, centralities





    def clustering_uncertainty_cal(self, local_regions, labeling_result, n_views, n_samples):

        # 初始化属性，清空旧数据
        uncertainties_dicts = []

        # 防御性检查：未选中节点列表为空
        if not self.unselected_nodes:
            print("警告：未选中节点列表（unselected_nodes）为空，无需计算聚类不确定性！")


        # ========== 第一步：遍历每个视图，仅计算未选中节点的聚类不确定性 ==========
        for view_idx in range(n_views):
            # print(f"\n======== 计算视角 {view_idx} 未选中节点的聚类不确定性 ========")
            # 直接使用 refine 生成的当前视图的聚类标签
            view_cluster_labels = labeling_result[view_idx]
            # 拿到当前视图的KNN局部区域
            current_local_regions = local_regions[view_idx]
            # 初始化当前视图的不确定性字典（仅存储未选中节点）
            view_uncertainty = {}
            # 仅遍历未选中节点（核心修改）
            for node in self.unselected_nodes:
                # 跳过超出范围的节点（防御性处理）
                if node >= n_samples:
                    print(f"警告：节点{node}超出样本总数{n_samples}，跳过")
                    continue
                # 拿到当前节点的KNN局部区域（近邻节点列表）
                knn_neighbors = current_local_regions[node]
                # k = len(knn_neighbors)  # 近邻数c
                # 统计局部区域内每个簇的样本数量
                cluster_count = defaultdict(int)
                for neighbor_node in knn_neighbors:
                    # 直接使用 refine 生成的标签
                    neighbor_cluster = view_cluster_labels.get(neighbor_node, -1)
                    # 跳过无标签的节点
                    if neighbor_cluster == -1:
                        continue
                    cluster_count[neighbor_cluster] += 1

                # 按公式计算香农熵（聚类不确定性）
                entropy = 0.0
                # 只有当局部区域有有效标签时才计算
                if sum(cluster_count.values()) > 0:
                    for cluster_id in cluster_count:
                        total_valid = sum(cluster_count.values())
                        p = cluster_count[cluster_id] / total_valid
                        if p > 0:
                            entropy -= p * np.log2(p)

                # 仅保存未选中节点的不确定性
                view_uncertainty[node] = entropy
            # print(entropy)

            # ========== 第二步：仅对未选中节点按不确定性从大到小排序 ==========
            def sort_key(item):
                node, uncertainty = item
                return -uncertainty  # 降序排列，不确定性越高越靠前
            sorted_items = sorted(view_uncertainty.items(), key=sort_key)
            sorted_node_dict = dict(sorted_items)
            uncertainties_dicts.append(sorted_node_dict)

        return uncertainties_dicts