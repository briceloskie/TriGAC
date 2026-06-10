import numpy as np
from scipy.io import loadmat
from typing import Tuple, List
import os


def load_multiview_dataset(
        dataset_path: str,
        return_original: bool = True
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    加载多视图数据集（不做任何预处理）

    参数:
        dataset_path: .mat 文件路径（绝对路径或相对路径）
        return_original: 是否返回原始格式（True 返回 n×d，False 返回 d×n）

    返回:
        X_list: 视图列表，每个元素 shape=(n_samples, d_v) 或 (d_v, n_samples)
        y: 标签数组，shape=(n_samples,)
    """
    print(f"正在加载数据集: {os.path.basename(dataset_path)}")
    print("=" * 60)

    # 1. 加载 mat 文件
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    mat_data = loadmat(dataset_path)

    # 2. 提取多视图数据（支持多种变量名格式）
    X_list = []

    # 优先查找标准变量名
    if 'X' in mat_data:
        X_cell = mat_data['X']
    elif 'fea' in mat_data:
        X_cell = mat_data['fea']
    elif 'data' in mat_data:
        X_cell = mat_data['data']
    elif 'Xs' in mat_data:
        X_cell = mat_data['Xs']
    else:
        # 自动搜索包含多视图数据的变量
        print("未找到标准数据变量，尝试自动识别...")
        found = False
        for key in mat_data.keys():
            if not key.startswith('_'):  # 过滤 MATLAB 内部变量
                value = mat_data[key]
                if isinstance(value, np.ndarray) and value.dtype == 'O':  # cell 数组
                    if value.shape[0] > 1 or value.shape[1] > 1:
                        X_cell = value
                        print(f"  自动识别数据变量: '{key}'")
                        found = True
                        break
        if not found:
            raise KeyError("无法找到多视图数据变量！支持的键名: X/fea/data/Xs 或 cell 数组")

    # 统一转换为 (视图数, 1) 格式
    if X_cell.shape[0] == 1:
        X_cell = X_cell.T

    n_views = X_cell.shape[0]
    print(f"  视图数量: {n_views}")

    # 3. 提取每个视图的数据
    for v in range(n_views):
        X_v = X_cell[v, 0]

        # 确保是二维数组
        if X_v.ndim == 1:
            X_v = X_v.reshape(-1, 1)

        # 可选：转置为 d×n 格式（某些算法需要）
        if not return_original:
            X_v = X_v.T

        X_list.append(X_v)
        print(f"  视图 {v + 1}: {X_v.shape}")

    # 4. 提取标签（支持多种变量名）
    label_keys = ['y', 'Y', 'gnd', 'GND', 'label', 'Label', 'gt', 'class', 'true_label']
    y = None

    for key in label_keys:
        if key in mat_data:
            y = mat_data[key].flatten()
            print(f"  标签变量: '{key}'")
            break

    if y is None:
        # 尝试自动识别
        print("  未找到标准标签变量，尝试自动识别...")
        for key in mat_data.keys():
            if not key.startswith('_') and key not in ['X', 'fea', 'data', 'Xs']:
                value = mat_data[key]
                if isinstance(value, np.ndarray) and value.ndim <= 2:
                    if value.shape[0] == X_list[0].shape[0] or value.shape[0] == X_list[0].shape[1]:
                        y = value.flatten()
                        print(f"    自动识别标签变量: '{key}'")
                        break

        if y is None:
            raise KeyError("无法找到标签变量！支持的键名: y/Y/gnd/label/gt/class")

    # 5. 确保标签从 1 开始
    if np.min(y) <= 0:
        y = y - np.min(y) + 1

    n_samples = len(y)
    n_classes = len(np.unique(y))

    print(f"  样本数量: {n_samples}")
    print(f"  类别数量: {n_classes}")
    print("=" * 60)

    return X_list, y


def load_dataset_by_name(
        dataset_name: str,
        base_dir: str = None,
        return_original: bool = True
) -> Tuple[List[np.ndarray], np.ndarray]:
    # 构建数据集路径
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, dataset_name)
    mat_file = os.path.join(base_dir, f"{dataset_name}.mat")
    if not os.path.exists(mat_file):
        mat_file = os.path.join(dataset_dir, f"{dataset_name}.mat")


    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"数据集文件不存在: {mat_file}\n请检查数据集名称是否正确")

    return load_multiview_dataset(mat_file, return_original)






