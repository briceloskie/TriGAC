import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_blobs

from main_realworld import preprocess_multiview_data, run_experiment


def generate_synthetic_multiview_dataset(
        n_samples=1000,
        n_views=5,
        n_clusters=10,
        n_features_per_view=20,
        random_state=42,
        noise_std=0.5,
        outlier_fraction=0.0,
        noisy_view_indices=None,
        noisy_view_noise_factor=3.0
):
    """
    生成合成多视角数据集 (支持多种噪音注入)

    参数:
        n_samples: 样本数量
        n_views: 视图数量
        n_clusters: 簇/类别数量
        n_features_per_view: 每个视图的特征维度
        random_state: 随机种子
        noise_std: 基础高斯噪音强度
        outlier_fraction: 离群点比例 (0.0 ~ 0.2)
        noisy_view_indices: 需要被污染的视图索引列表
        noisy_view_noise_factor: 低质量视图的噪音放大倍数
    """
    print("=" * 80)
    print("🔄 [步骤 1/3] 正在生成合成多视角数据集...")
    print("=" * 80)
    print(f"  样本数量: {n_samples:,}")
    print(f"  视图数量: {n_views}")
    print(f"  类别数量: {n_clusters}")
    print(f"  每视图维度: {n_features_per_view}")
    print(f"  [噪音] 基础高斯噪音 Std: {noise_std}")
    print(f"  [噪音] 离群点比例: {outlier_fraction:.2%}")
    if noisy_view_indices:
        print(f"  [噪音] 低质量视图 ID: {noisy_view_indices}, 噪音放大: {noisy_view_noise_factor}x")
    print("=" * 80)

    if noisy_view_indices is None:
        noisy_view_indices = []

    X_list = []
    rng = np.random.RandomState(random_state)

    print("\n  ⏳ 正在生成基础标签和中心点...", end='', flush=True)
    X_base, y, centers = make_blobs(
        n_samples=n_samples,
        n_features=n_features_per_view,
        centers=n_clusters,
        random_state=random_state,
        return_centers=True
    )
    print(" ✓")

    n_outliers = int(n_samples * outlier_fraction)
    is_outlier = np.zeros(n_samples, dtype=bool)
    if n_outliers > 0:
        print(f"  ⏳ 正在生成 {n_outliers} 个离群点...", end='', flush=True)
        outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
        is_outlier[outlier_indices] = True
        print(" ✓")

    data_min, data_max = centers.min(), centers.max()
    padding = (data_max - data_min) * 2.0

    for view_idx in range(n_views):
        print(f"\n  📊 生成视图 {view_idx + 1}/{n_views}...", end='', flush=True)
        np.random.seed(random_state + view_idx * 100)

        view_centers = centers + np.random.randn(n_clusters, n_features_per_view) * 2

        current_noise_std = noise_std
        is_noisy_view = view_idx in noisy_view_indices
        if is_noisy_view:
            current_noise_std *= noisy_view_noise_factor

        X_view = np.zeros((n_samples, n_features_per_view))

        for cluster_id in range(n_clusters):
            mask = (y == cluster_id) & (~is_outlier)
            n_cluster_samples = mask.sum()

            if n_cluster_samples > 0:
                noise = np.random.randn(n_cluster_samples, n_features_per_view) * current_noise_std
                X_view[mask] = view_centers[cluster_id] + noise

        if n_outliers > 0:
            outliers = rng.uniform(data_min - padding, data_max + padding, (n_outliers, n_features_per_view))
            outlier_noise = rng.randn(n_outliers, n_features_per_view) * (current_noise_std * 2)
            X_view[is_outlier] = outliers + outlier_noise

        X_list.append(X_view)

        status = " (⚠️ 低质量视图)" if is_noisy_view else ""
        print(f" ✓{status}")

    print(f"\n✓ 合成数据集生成完成！")
    print(f"  标签分布: {np.bincount(y)}")
    print(f"  类别数: {n_clusters}")

    return X_list, y


if __name__ == '__main__':
    # ===================== 配置区域 =====================
    n_iterations = 20
    window_size = 500
    high_dim_threshold = 50
    umap_n_components = 20

    # 合成数据集参数
    synthetic_config = {
        'n_samples': 1000,
        'n_views': 5,
        'n_clusters': 5,
        'n_features_per_view': 5,
        'random_state': 42,
        'noise_std': 0.8,
        'outlier_fraction': 0.05,
        'noisy_view_indices': [2],
        'noisy_view_noise_factor': 3.0
    }

    # ===================== 运行合成大数据集实验 =====================
    print(f"\n{'=' * 80}")
    print(f"开始处理合成大数据集 (Synthetic Big Data)")
    print(f"{'=' * 80}")

    try:
        X_list_syn, y_syn = generate_synthetic_multiview_dataset(**synthetic_config)

        print(f"\n  开始合成数据预处理...")
        X_list_syn_processed = preprocess_multiview_data(
            X_list_syn,
            high_dim_threshold=high_dim_threshold,
            umap_n_components=umap_n_components
        )

        result_df_syn = run_experiment(
            X_list_original=X_list_syn_processed,
            y=y_syn,
            k=synthetic_config['n_clusters'],
            n_iterations=n_iterations,
            window_size=window_size
        )

        results_dir = "results_all_datasets_select"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        csv_filename = "Synthetic_BigData_results.csv"
        csv_save_path = os.path.join(results_dir, csv_filename)
        result_df_syn.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

        final_ari = result_df_syn['ARI'].iloc[-1]
        final_nmi = result_df_syn['NMI'].iloc[-1]
        print(f"\n  ✅ 合成数据实验完成 | 最终 ARI: {final_ari:.4f}, NMI: {final_nmi:.4f}")
        print(f"     结果已保存至: {csv_save_path}")

    except Exception as e:
        print(f"\n❌ 合成数据实验失败: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 80}")
