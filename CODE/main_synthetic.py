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
    Generate synthetic multi-view dataset (with multiple noise injection)

    Parameters:
        n_samples: Number of samples
        n_views: Number of views
        n_clusters: Number of clusters/classes
        n_features_per_view: Feature dimension per view
        random_state: Random seed
        noise_std: Base Gaussian noise intensity
        outlier_fraction: Outlier proportion (0.0 ~ 0.2)
        noisy_view_indices: List of view indices to be corrupted
        noisy_view_noise_factor: Noise amplification factor for low-quality views
    """
    print("=" * 80)
    print("🔄 [Step 1/3] Generating synthetic multi-view dataset...")
    print("=" * 80)
    print(f"  Number of samples: {n_samples:,}")
    print(f"  Number of views: {n_views}")
    print(f"  Number of classes: {n_clusters}")
    print(f"  Dimensions per view: {n_features_per_view}")
    print(f"  [Noise] Base Gaussian noise Std: {noise_std}")
    print(f"  [Noise] Outlier fraction: {outlier_fraction:.2%}")
    if noisy_view_indices:
        print(f"  [Noise] Low-quality view IDs: {noisy_view_indices}, Noise amplification: {noisy_view_noise_factor}x")
    print("=" * 80)

    if noisy_view_indices is None:
        noisy_view_indices = []

    X_list = []
    rng = np.random.RandomState(random_state)

    print("\n  ⏳ Generating base labels and centers...", end='', flush=True)
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
        print(f"  ⏳ Generating {n_outliers} outliers...", end='', flush=True)
        outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
        is_outlier[outlier_indices] = True
        print(" ✓")

    data_min, data_max = centers.min(), centers.max()
    padding = (data_max - data_min) * 2.0

    for view_idx in range(n_views):
        print(f"\n  📊 Generating view {view_idx + 1}/{n_views}...", end='', flush=True)
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

        status = " (⚠️ Low-quality view)" if is_noisy_view else ""
        print(f" ✓{status}")

    print(f"\n✓ Synthetic dataset generation completed!")
    print(f"  Label distribution: {np.bincount(y)}")
    print(f"  Number of classes: {n_clusters}")

    return X_list, y


if __name__ == '__main__':
    # ===================== Configuration Area =====================
    n_iterations = 20
    window_size = 500
    high_dim_threshold = 50
    umap_n_components = 20

    # Synthetic dataset parameters
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

    # ===================== Run Synthetic Big Data Experiment =====================
    print(f"\n{'=' * 80}")
    print(f"Starting to process Synthetic Big Data")
    print(f"{'=' * 80}")

    try:
        X_list_syn, y_syn = generate_synthetic_multiview_dataset(**synthetic_config)

        print(f"\n  Starting synthetic data preprocessing...")
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
        print(f"\n  ✅ Synthetic data experiment completed | Final ARI: {final_ari:.4f}, NMI: {final_nmi:.4f}")
        print(f"     Results saved to: {csv_save_path}")

    except Exception as e:
        print(f"\n❌ Synthetic data experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 80}")

