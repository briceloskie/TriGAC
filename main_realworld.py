import numpy as np
import pandas as pd
import os
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ActiveSelecter import ActiveSelecter
from GeneralUtils import GeneralUtils
from HumanAnnotator import HumanAnnotator
from NodeSkeletonBuilder import NodeSkeletonBuilder
from NodeSkeletonRefiner import NodeSkeletonRefiner
from WeightUpdator import WeightUpdator
from datasets.data_loader import load_dataset_by_name


def preprocess_multiview_data(X_list_original, high_dim_threshold=50, umap_n_components=20, random_state=42):
    """
    多视图数据完整预处理流程 (简化版：低维跳过降维)

    逻辑:
        - 如果原始维度 >= high_dim_threshold: Z-score -> PCA(50) -> UMAP(20) -> L2
        - 如果原始维度 < high_dim_threshold:  只做 Z-score -> L2 (跳过 PCA 和 UMAP)
    """
    X_list_processed = []

    for v_idx, X_v in enumerate(X_list_original):
        n_samples, n_features = X_v.shape
        print(f"\n  视图 {v_idx + 1} 预处理（原始维度: {n_features}）:")

        scaler = StandardScaler()
        X_v_processed = scaler.fit_transform(X_v)
        print(f"[1/2] Z-score 标准化完成，均值: {X_v_processed.mean():.4f}, 标准差: {X_v_processed.std():.4f}")

        if n_features > high_dim_threshold:
            print(f"[2/2] 维度 {n_features} > {high_dim_threshold}，执行 PCA + UMAP 降维")

            pca = PCA(n_components=high_dim_threshold, random_state=random_state)
            X_v_pca = pca.fit_transform(X_v_processed)
            explained_var = np.sum(pca.explained_variance_ratio_)
            print(f"      PCA 降至 {high_dim_threshold} 维完成，保留方差比例: {explained_var:.4f}")

            reducer = UMAP(
                n_components=umap_n_components,
                random_state=random_state,
                n_neighbors=20,
                min_dist=0.1,
                metric='euclidean'
            )
            X_v_processed = reducer.fit_transform(X_v_pca)
            print(f"      UMAP 降至 {umap_n_components} 维完成")

        elif n_features >= umap_n_components:
            print(f"[2/2] 维度 {n_features} 在 [{umap_n_components}, {high_dim_threshold}] 范围，仅执行 UMAP 降维")

            reducer = UMAP(
                n_components=umap_n_components,
                random_state=random_state,
                n_neighbors=20,
                min_dist=0.1,
                metric='euclidean'
            )
            X_v_processed = reducer.fit_transform(X_v_processed)
            print(f"      UMAP 降至 {umap_n_components} 维完成")

        else:
            print(f"[2/2] 维度 {n_features} < {umap_n_components}，跳过降维")

        l2_norms = np.linalg.norm(X_v_processed, axis=1, keepdims=True)
        l2_norms[l2_norms < 1e-8] = 1.0
        X_v_normalized = X_v_processed / l2_norms

        X_list_processed.append(X_v_normalized)

    return X_list_processed


def run_experiment(X_list_original, y, k, n_iterations, window_size, max_budget=None):
    n_samples = len(X_list_original[0])
    n_views = len(X_list_original)
    c = len(set(y))
    N = n_samples

    print(f"  样本数: {n_samples}, 视图数: {n_views}, 类别数: {c}")
    if max_budget is not None:
        print(f"  ⚙️  最大预算限制: max_budget={max_budget}")

    view_weights = np.ones(n_views) / n_views

    generalutils = GeneralUtils(n_samples, n_views)
    local_regions = generalutils._local_regions_search(X_list_original, k)

    builder = NodeSkeletonBuilder()
    node_skeletons, centralities_dicts = builder.build(X_list_original)

    selecter = ActiveSelecter()
    annotator = HumanAnnotator(X_list_original, y, window_size, max_budget=max_budget)
    refiner = NodeSkeletonRefiner()
    weightupdator = WeightUpdator(n_samples, n_views)
    result_records = []
    selected_num = 0

    print(f"\n📊 正在计算第 0 轮 (Baseline) 指标 (所有点归为一簇)...")
    baseline_pred = np.zeros(n_samples, dtype=int)

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari_baseline = adjusted_rand_score(y, baseline_pred)
        nmi_baseline = normalized_mutual_info_score(y, baseline_pred)
    except Exception as e:
        print(f"⚠️ 计算基线指标失败: {e}")
        ari_baseline, nmi_baseline = 0.0, 0.0

    result_records.append({
        "轮次": 0,
        "selected_num": 0,
        "count": 0,
        "ARI": round(ari_baseline, 4),
        "NMI": round(nmi_baseline, 4)
    })
    print(f"   -> Baseline: count=0 | ARI: {ari_baseline:.4f} | NMI: {nmi_baseline:.4f}")

    for iteration in range(1, n_iterations + 1):
        budget = int(N / n_iterations)
        remaining_nodes = len(selecter.unselected_nodes)

        if 0 < remaining_nodes <= budget:
            budget = remaining_nodes
            print(f"\n🏁 [扫尾模式] 剩余节点数({remaining_nodes}) <= 预算，将执行最后一轮扫尾...")

        elif remaining_nodes > 0:
            if iteration == n_iterations:
                budget = max(budget, remaining_nodes)

        print(f"\n{'=' * 60}")
        print(f"开始第 {iteration}/{n_iterations} 轮迭代 | 当前剩余未选节点: {remaining_nodes} | 本轮预算: {budget}")
        print(f"{'=' * 60}")

        if iteration == 1:
            selected_nodes, centralities_dicts = selecter.node_selection_first_round(
                centralities_dicts, budget, n_views, n_samples
            )
        else:
            uncertainties_dicts = selecter.clustering_uncertainty_cal(
                local_regions, refiner.labeling_result, n_views, n_samples
            )
            selected_nodes, centralities_dicts = selecter.node_selection_subsequent_round(
                budget, n_views, view_weights, uncertainties_dicts, centralities_dicts
            )

        if len(selected_nodes) == 0 and remaining_nodes > 0:
            print(f"\n⚠️ 警告：自动选点返回 0 个节点！强制从 unselected_nodes 中选取剩余 {remaining_nodes} 个节点。")
            selected_nodes = selecter.unselected_nodes.copy()
            for node in selected_nodes:
                for d in centralities_dicts:
                    d.pop(node, None)
            selecter.unselected_nodes = []

        print(f"   实际选中节点数: {len(selected_nodes)}")
        annotator._pairwise_contraint_annotation(node_skeletons, selected_nodes, view_weights)
        selected_num = selected_num + len(selected_nodes)

        node_skeletons = refiner.refine(node_skeletons, annotator.neighborhoods, view_weights)

        view_weights = weightupdator.update_weights(
            refiner.labeling_result,
            refiner.node_final_neigh,
            method='nmi'
        )

        ari, nmi, pred_labels = generalutils._refined_result_to_metrics(
            refined_result=refiner.node_final_neigh, y=y
        )
        result_record = {
            "轮次": iteration,
            "selected_num": selected_num,
            "count": annotator.count,
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4)
        }
        result_records.append(result_record)

        print(
            f"    轮次 {iteration}/{n_iterations} | count: {annotator.count} | selected_num: {selected_num}/{N} | ARI: {ari:.4f} | NMI: {nmi:.4f}")

        if selected_num >= N:
            print(f"\n✅ 所有 {N} 个节点已选完，实验提前结束。")
            break

        if annotator.budget_reached:
            print(f"\n🛑 已达到最大预算限制 (max_budget={max_budget})，实验提前终止")
            print(f"   最终统计：count={annotator.count}, selected_num={selected_num}, ARI={ari:.4f}, NMI={nmi:.4f}")
            break

    result_df = pd.DataFrame(result_records)
    return result_df


if __name__ == '__main__':
    # ===================== 配置区域 =====================
    datasets_config = {
        'BBCSport': lambda: load_dataset_by_name('BBCSport'),
    }

    k = 15
    n_iterations = 20
    window_size = 500
    max_val = 350
    budget_sequence = [int(0.2 * max_val), int(0.4 * max_val), int(0.6 * max_val), int(0.8 * max_val), int(max_val)]
    high_dim_threshold = 50
    umap_n_components = 20

    # ===================== 遍历所有数据集 =====================
    for dataset_name, data_loader in datasets_config.items():
        try:
            print(f"\n{'=' * 80}")
            print(f"开始处理数据集: {dataset_name}")
            print(f"{'=' * 80}")

            X_list_original, y = data_loader()

            print(f"\n  开始数据预处理（流程: Z-score → PCA(>{high_dim_threshold}) → UMAP({umap_n_components}) → L2）:")
            X_list_processed = preprocess_multiview_data(
                X_list_original,
                high_dim_threshold=high_dim_threshold,
                umap_n_components=umap_n_components
            )

            budget_results_summary = []

            for max_budget in budget_sequence:
                print(f"\n{'#' * 60}")
                print(f"🔬 开始执行实验 | 数据集: {dataset_name} | 预算: {max_budget}")
                print(f"{'#' * 60}")

                result_df = run_experiment(
                    X_list_original=X_list_processed,
                    y=y,
                    k=k,
                    n_iterations=n_iterations,
                    window_size=window_size,
                    max_budget=max_budget
                )

                final_row = result_df.iloc[-1]
                budget_result = {
                    "max_budget": max_budget,
                    "selected_num": int(final_row['selected_num']),
                    "count": int(final_row['count']),
                    "ARI": float(final_row['ARI']),
                    "NMI": float(final_row['NMI'])
                }
                budget_results_summary.append(budget_result)

                print(
                    f"\n✅ 预算 {max_budget} 实验完成 | ARI: {budget_result['ARI']:.4f}, NMI: {budget_result['NMI']:.4f}")

            results_dir = f"results_all_datasets_finetunned"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            summary_df = pd.DataFrame(budget_results_summary)
            csv_filename = f"{dataset_name}_budget_sequence_results.csv"
            csv_save_path = os.path.join(results_dir, csv_filename)
            summary_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

            print(f"\n{'=' * 80}")
            print(f"📊 {dataset_name} 预算序列实验汇总:")
            print(summary_df.to_string(index=False))
            print(f"\n💾 汇总结果已保存至: {csv_save_path}")
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"\n❌ {dataset_name} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print(f"所有真实数据集处理完成！")
    print(f"{'=' * 80}")
    print(f"\n结果保存在以下目录:")
    print(f"  - ./results_all_datasets_finetunned")
