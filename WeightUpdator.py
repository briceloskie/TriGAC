import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from ViewSkeletonBuilder import ViewSkeletonBuilder


class WeightUpdator:
    def __init__(self, n_samples, n_views):
        self.n_samples = n_samples
        self.n_views = n_views

    def _calculate_view_similarity_matrix(self, labeling_result, method='nmi'):
        """
        计算视图间的相似度矩阵

        :param labeling_result: list of dict, 每个视图的节点-邻域标签映射 [{node_id: label}, ...]
        :param method: str, 相似度计算方法 'nmi' 或 'ari'
        :return: np.array, 相似度矩阵 (n_views, n_views)，对称矩阵，对角线为1
        """
        n_views = len(labeling_result)
        similarity_matrix = np.zeros((n_views, n_views))

        # 提取所有视图的标签数组
        view_labels_list = []
        for view_idx in range(n_views):
            view_node2label = labeling_result[view_idx]
            view_labels = np.array([view_node2label.get(node_id, -1) for node_id in range(self.n_samples)])
            view_labels_list.append(view_labels)

        # 计算每对视图之间的相似度
        for i in range(n_views):
            for j in range(i, n_views):
                if i == j:
                    # 对角线：视图与自身相似度为1
                    similarity_matrix[i, j] = 1.0
                else:
                    # 获取两个视图的标签
                    labels_i = view_labels_list[i]
                    labels_j = view_labels_list[j]

                    # 取两个视图都有效的节点
                    valid_mask = (labels_i != -1) & (labels_j != -1)

                    if not np.any(valid_mask):
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
                        continue

                    labels_i_valid = labels_i[valid_mask]
                    labels_j_valid = labels_j[valid_mask]

                    # 检查是否有足够的唯一标签
                    if len(np.unique(labels_i_valid)) < 2 or len(np.unique(labels_j_valid)) < 2:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
                        continue

                    # 计算相似度
                    try:
                        if method == 'nmi':
                            sim = normalized_mutual_info_score(labels_i_valid, labels_j_valid)
                        elif method == 'ari':
                            sim = adjusted_rand_score(labels_i_valid, labels_j_valid)
                        else:
                            raise ValueError(f"不支持的相似度方法: {method}")

                        # 对称矩阵
                        similarity_matrix[i, j] = max(sim, 0.0)
                        similarity_matrix[j, i] = max(sim, 0.0)
                    except Exception as e:
                        print(f"⚠️ 视图{i}与视图{j} 相似度计算失败: {e}")
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0

        return similarity_matrix

    def _calculate_view_quality(self, labeling_result, node_final_neigh, method='nmi'):
        """
        计算单个视图的质量 q_v：视图骨架与融合骨架的结构相似度

        :param labeling_result: list of dict, 每个视图的节点-邻域标签映射
        :param node_final_neigh: dict, 融合后的节点-邻域标签映射
        :param method: str, 相似度计算方法 'nmi' 或 'ari'
        :return: np.array, 每个视图的质量得分 (n_views,)
        """
        n_views = len(labeling_result)
        view_qualities = np.zeros(n_views)

        # 提取融合结果的标签数组
        fused_labels = np.array([node_final_neigh.get(node_id, -1) for node_id in range(self.n_samples)])
        valid_mask = fused_labels != -1

        if not np.any(valid_mask):
            print("⚠️ 警告：融合结果中无有效标签，返回均匀质量")
            return np.ones(n_views) / n_views

        fused_labels_valid = fused_labels[valid_mask]

        # 计算每个视图与融合结果的相似度作为质量
        for view_idx in range(n_views):
            view_node2label = labeling_result[view_idx]
            view_labels = np.array([view_node2label.get(node_id, -1) for node_id in range(self.n_samples)])
            view_labels_valid = view_labels[valid_mask]

            # 过滤视图中无标签的节点
            view_valid_mask = view_labels_valid != -1
            if not np.any(view_valid_mask):
                view_qualities[view_idx] = 0.0
                continue

            # 取两者共同有效的节点
            common_valid_mask = view_valid_mask
            fused_common = fused_labels_valid[common_valid_mask]
            view_common = view_labels_valid[common_valid_mask]

            # 检查是否有足够的唯一标签
            if len(np.unique(fused_common)) < 2 or len(np.unique(view_common)) < 2:
                view_qualities[view_idx] = 0.0
                continue

            # 计算相似度作为质量
            try:
                if method == 'nmi':
                    quality = normalized_mutual_info_score(fused_common, view_common)
                elif method == 'ari':
                    quality = adjusted_rand_score(fused_common, view_common)
                else:
                    raise ValueError(f"不支持的相似度方法: {method}")

                view_qualities[view_idx] = max(quality, 0.0)
            except Exception as e:
                print(f"⚠️ 视图{view_idx} 质量计算失败: {e}")
                view_qualities[view_idx] = 0.0

        return view_qualities

    def update_weights(self, labeling_result, node_final_neigh, method='nmi', n_clusters=None):
        """
        层次化质量驱动的视图权重更新（三步走策略）

        步骤1：基于视图相似度矩阵聚类，得到 K 个簇
        步骤2：给每个簇分配总权重配额（与簇的整体质量成正比）
        步骤3：簇内权重分配（与视图个体质量成正比）

        :param labeling_result: list of dict, 每个视图的节点-邻域标签映射
        :param node_final_neigh: dict, 融合后的节点-邻域标签映射
        :param method: str, 相似度计算方法 'nmi' 或 'ari'
        :param n_clusters: int, 期望的簇数量（None则自动确定）
        :return: np.array, 更新后的视图权重 (n_views,)，权重和为1
        """
        n_views = len(labeling_result)

        # ===================== 步骤1：视图聚类 =====================
        # print("\n📌 步骤1：基于视图相似度矩阵进行聚类")

        # 计算视图间的相似度矩阵
        view_similarity_matrix = self._calculate_view_similarity_matrix(labeling_result, method=method)
        # print(f"视图间相似度矩阵:\n{np.round(view_similarity_matrix, 4)}")

        # 使用 ViewSkeletonBuilder 构建视图骨架并分簇
        viewbuilder = ViewSkeletonBuilder()
        view_skeleton = viewbuilder.build(view_similarity_matrix)
        view_clusters = viewbuilder.clusters_division(view_skeleton)

        K = len(view_clusters)
        # print(f"✅ 视图聚类完成：{n_views} 个视图 → {K} 个簇")
        # for k, cluster in enumerate(view_clusters):
        #     print(f"  簇{k}: 视图 {cluster}")

        # ===================== 步骤2：计算视图质量 =====================
        # print("\n📌 步骤2：计算每个视图的质量得分")
        view_qualities = self._calculate_view_quality(labeling_result, node_final_neigh, method=method)
        # print(f"视图质量得分: {[round(q, 4) for q in view_qualities]}")

        # ===================== 步骤3：簇级配额分配 =====================
        # print("\n📌 步骤3：基于簇质量分配总权重配额")
        cluster_weights = np.zeros(K)

        for k, cluster in enumerate(view_clusters):
            # 簇的整体质量 = 簇内视图的平均质量
            cluster_quality = np.mean([view_qualities[v] for v in cluster])
            cluster_weights[k] = cluster_quality

        # 归一化：确保所有簇的配额加起来 = 1
        total_cluster_weight = np.sum(cluster_weights)
        if total_cluster_weight < 1e-12:
            # 所有簇质量为0，均匀分配
            cluster_weights = np.ones(K) / K
        else:
            cluster_weights = cluster_weights / total_cluster_weight

        # print(f"簇级权重配额: {[round(w, 4) for w in cluster_weights]}")

        # ===================== 步骤4：簇内权重分配 =====================
        # print("\n📌 步骤4：簇内权重分配（基于个体质量占比）")
        final_weights = np.zeros(n_views)

        for k, cluster in enumerate(view_clusters):
            # 簇内视图的质量列表
            cluster_qualities = np.array([view_qualities[v] for v in cluster])
            total_quality = np.sum(cluster_qualities)

            if total_quality < 1e-12:
                # 簇内所有视图质量为0，均分簇配额
                for v in cluster:
                    final_weights[v] = cluster_weights[k] / len(cluster)
            else:
                # 按质量占比分配簇配额
                for idx, v in enumerate(cluster):
                    final_weights[v] = cluster_weights[k] * (cluster_qualities[idx] / total_quality)

        # print(f"✅ 最终视图权重（和为{np.sum(final_weights):.4f}）: {[round(w, 4) for w in final_weights]}")

        return final_weights
