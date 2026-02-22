from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DEFAULT_TERRAIN_COSTS: dict[str, int] = {
	"Black": 999,
	"Green": 3,
	"Blue": 10,
	"Grey": 1,
	"White": 1,
	"Yellow": 5,
	"Brown": 4,
	"Orange": 6,
	"Red": 8,
	"Pink": 7,
	"Purple": 6,
}


def load_color_dataset(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
	rows_r: list[float] = []
	rows_g: list[float] = []
	rows_b: list[float] = []
	labels: list[str] = []

	dataset_path = Path(csv_path)
	with dataset_path.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		required_columns = {"red", "green", "blue", "label"}
		if reader.fieldnames is None or not required_columns.issubset({name.strip().lower() for name in reader.fieldnames}):
			raise ValueError("CSV inválido. Se esperaban columnas: red, green, blue, label")

		for row in reader:
			rows_r.append(float(row["red"]))
			rows_g.append(float(row["green"]))
			rows_b.append(float(row["blue"]))
			labels.append(str(row["label"]).strip())

	x = np.column_stack([rows_r, rows_g, rows_b]).astype(np.float64)
	x = x / 255.0

	unique_labels = sorted(set(labels))
	label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
	y = np.array([label_to_index[label] for label in labels], dtype=np.int64)

	return x, y, unique_labels, label_to_index


def split_train_test(
	x: np.ndarray,
	y: np.ndarray,
	test_size: float = 0.2,
	seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if not 0 < test_size < 1:
		raise ValueError("test_size debe estar en (0,1)")
	if x.shape[0] != y.shape[0]:
		raise ValueError("x e y deben tener la misma cantidad de filas")

	rng = np.random.default_rng(seed)
	indices = np.arange(x.shape[0])
	rng.shuffle(indices)

	test_count = int(round(x.shape[0] * test_size))
	test_indices = indices[:test_count]
	train_indices = indices[test_count:]

	return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def build_terrain_cost_map(
	labels: list[str],
	custom_costs: dict[str, int] | None = None,
	default_cost: int = 5,
) -> dict[str, int]:
	costs = dict(DEFAULT_TERRAIN_COSTS)
	if custom_costs:
		costs.update(custom_costs)

	return {label: costs.get(label, default_cost) for label in labels}
