import numpy as np
from scipy.io import loadmat
from typing import Tuple, List
import os


def load_multiview_dataset(
        dataset_path: str,
        return_original: bool = True
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load multi-view dataset (without any preprocessing)

    Parameters:
        dataset_path: .mat file path (absolute or relative path)
        return_original: whether to return original format (True returns n×d, False returns d×n)

    Returns:
        X_list: view list, each element shape=(n_samples, d_v) or (d_v, n_samples)
        y: label array, shape=(n_samples,)
    """
    print(f"Loading dataset: {os.path.basename(dataset_path)}")
    print("=" * 60)

    # 1. Load mat file
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    mat_data = loadmat(dataset_path)

    # 2. Extract multi-view data (support multiple variable name formats)
    X_list = []

    # Priority search for standard variable names
    if 'X' in mat_data:
        X_cell = mat_data['X']
    elif 'fea' in mat_data:
        X_cell = mat_data['fea']
    elif 'data' in mat_data:
        X_cell = mat_data['data']
    elif 'Xs' in mat_data:
        X_cell = mat_data['Xs']
    else:
        # Automatically search for variables containing multi-view data
        print("Standard data variable not found, attempting automatic recognition...")
        found = False
        for key in mat_data.keys():
            if not key.startswith('_'):  # Filter MATLAB internal variables
                value = mat_data[key]
                if isinstance(value, np.ndarray) and value.dtype == 'O':  # cell array
                    if value.shape[0] > 1 or value.shape[1] > 1:
                        X_cell = value
                        print(f"  Automatically recognized data variable: '{key}'")
                        found = True
                        break
        if not found:
            raise KeyError("Cannot find multi-view data variable! Supported keys: X/fea/data/Xs or cell array")

    # Convert to (num_views, 1) format uniformly
    if X_cell.shape[0] == 1:
        X_cell = X_cell.T

    n_views = X_cell.shape[0]
    print(f"  Number of views: {n_views}")

    # 3. Extract data for each view
    for v in range(n_views):
        X_v = X_cell[v, 0]

        # Ensure it is a 2D array
        if X_v.ndim == 1:
            X_v = X_v.reshape(-1, 1)

        # Optional: transpose to d×n format (required by some algorithms)
        if not return_original:
            X_v = X_v.T

        X_list.append(X_v)
        print(f"  View {v + 1}: {X_v.shape}")

    # 4. Extract labels (support multiple variable names)
    label_keys = ['y', 'Y', 'gnd', 'GND', 'label', 'Label', 'gt', 'class', 'true_label']
    y = None

    for key in label_keys:
        if key in mat_data:
            y = mat_data[key].flatten()
            print(f"  Label variable: '{key}'")
            break

    if y is None:
        # Try automatic recognition
        print("  Standard label variable not found, attempting automatic recognition...")
        for key in mat_data.keys():
            if not key.startswith('_') and key not in ['X', 'fea', 'data', 'Xs']:
                value = mat_data[key]
                if isinstance(value, np.ndarray) and value.ndim <= 2:
                    if value.shape[0] == X_list[0].shape[0] or value.shape[0] == X_list[0].shape[1]:
                        y = value.flatten()
                        print(f"    Automatically recognized label variable: '{key}'")
                        break

        if y is None:
            raise KeyError("Cannot find label variable! Supported keys: y/Y/gnd/label/gt/class")

    # 5. Ensure labels start from 1
    if np.min(y) <= 0:
        y = y - np.min(y) + 1

    n_samples = len(y)
    n_classes = len(np.unique(y))

    print(f"  Number of samples: {n_samples}")
    print(f"  Number of classes: {n_classes}")
    print("=" * 60)

    return X_list, y


def load_dataset_by_name(
        dataset_name: str,
        base_dir: str = None,
        return_original: bool = True
) -> Tuple[List[np.ndarray], np.ndarray]:
    # Build dataset path
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, dataset_name)
    mat_file = os.path.join(base_dir, f"{dataset_name}.mat")
    if not os.path.exists(mat_file):
        mat_file = os.path.join(dataset_dir, f"{dataset_name}.mat")


    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Dataset file does not exist: {mat_file}\nPlease check if the dataset name is correct")

    return load_multiview_dataset(mat_file, return_original)
