"""Eleven small datasets and the five independent manuscript query caps.

The keys match filenames in ``datasets/``.  ``Animal.mat`` is intentionally
not part of this 11-dataset experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    display_name: str
    loader_name: str
    budgets: tuple[int, int, int, int, int]


DATASET_CONFIGS = {
    "Sources3": DatasetConfig("3Sources", "Sources3", (30, 60, 90, 120, 150)),
    "BBCSport": DatasetConfig("BBCSport", "BBCSport", (50, 100, 150, 200, 250)),
    "Movies": DatasetConfig("Movies", "Movies", (600, 1200, 1800, 2400, 3000)),
    "WebKB": DatasetConfig("WebKB", "WebKB", (200, 400, 600, 800, 1000)),
    "Cal5": DatasetConfig("Cal-5V", "Cal5", (250, 500, 750, 1000, 1250)),
    "RGBD": DatasetConfig("RGBD", "RGBD", (700, 1400, 2100, 2800, 3500)),
    "Youtube": DatasetConfig("YouTube", "Youtube", (1000, 2000, 3000, 4000, 5000)),
    "Cal20": DatasetConfig("Cal20", "Cal20", (600, 1200, 1800, 2400, 3000)),
    "Cora": DatasetConfig("Cora", "Cora", (1000, 2000, 3000, 4000, 5000)),
    "CiteSeer": DatasetConfig("CiteSeer", "CiteSeer", (1400, 2800, 4200, 5600, 7000)),
    "MNIST4": DatasetConfig("MNIST-4", "MNIST4", (900, 1800, 2700, 3600, 4500)),
}


SMALL_DATASETS = tuple(DATASET_CONFIGS)
