import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import BallTree


class GeneralUtils:
    def __init__(self,n_samples,n_views):
        self.n_samples=n_samples
        self.n_views=n_views


    def _local_regions_search(self, X_list, k):
        n_samples = self.n_samples
        # 存储结果：neighbors_per_view[view_idx][sample_idx] = [k 个近邻的索引]
        local_regions = []

        for view_idx, X in enumerate(X_list):
            # print(f"\n处理视角 {view_idx + 1}/{n_views}...")

            # 使用 Ball Tree 加速近邻搜索
            tree = BallTree(X)
            # 对每个样本，查询 k 个最近邻（包括自己）
            # query 返回 (距离，索引)，k=k
            distances, indices = tree.query(X, k)

            # 将其转为字典格式：{样本索引：[近邻索引列表]}
            neighbors_this_view = {}
            for i in range(n_samples):
                neighbor_indices = indices[i].tolist()
                neighbors_this_view[i] = neighbor_indices

            local_regions.append(neighbors_this_view)
        return local_regions



    def _refined_result_to_metrics(self,refined_result, y):
        # ========== 计算ARI和NMI ==========
        # 步骤1：将refined_result转换为与真实标签y长度一致的数组（按节点ID排序）
        # 确保节点ID从0到n_samples-1，避免缺失
        n_samples = len(y)
        pred_labels = []
        for node_id in range(n_samples):
            pred_labels.append(refined_result.get(node_id, -1))  # 无标签的节点归为-1
        pred_labels = np.array(pred_labels)

        # 步骤2：过滤掉无标签的节点（可选，若有-1标签）
        valid_mask = pred_labels != -1
        y_valid = y[valid_mask]
        pred_valid = pred_labels[valid_mask]

        # 步骤3：计算ARI和NMI
        if len(y_valid) == 0:
            print("⚠️ 无有效标签节点，无法计算评估指标！")
            ari = np.nan
            nmi = np.nan
        else:
            ari = adjusted_rand_score(y_valid, pred_valid)
            nmi = normalized_mutual_info_score(y_valid, pred_valid)

        # 步骤4：打印评估结果
        # print("\n======== 聚类评估结果 ========")
        # print(f"有效节点数：{len(y_valid)} / {n_samples}")
        # print(f"调整兰德指数（ARI）: {ari:.4f}")  # 越接近1越好，0为随机聚类
        # print(f"归一化互信息（NMI）: {nmi:.4f}")  # 越接近1越好，0为无关联

        # 关键修改：返回ARI和NMI
        return ari, nmi,pred_labels


