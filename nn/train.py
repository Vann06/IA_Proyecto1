from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nn.dataset import build_terrain_cost_map, load_color_dataset, split_train_test
from nn.mlp import MLPClassifierNumpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "assets" / "final_data_colors.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Entrena un MLP en numpy para clasificar colores RGB.")
	parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Ruta al CSV de colores")
	parser.add_argument("--epochs", type=int, default=120, help="Número de épocas")
	parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate SGD")
	parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de mini-batch")
	parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
	parser.add_argument(
		"--hidden-layers",
		type=int,
		nargs="+",
		default=[16, 12],
		help="Neuronas por capa oculta. Ej: --hidden-layers 24 16",
	)
	return parser


def _write_report(
	report_path: Path,
	labels: list[str],
	terrain_cost_map: dict[str, int],
	test_accuracy: float,
	train_accuracy: float,
	epochs: int,
	hidden_layers: tuple[int, ...],
) -> None:
	lines: list[str] = []
	lines.append("Task 2.1 - Entrenamiento de Red Neuronal (MLP numpy)")
	lines.append("====================================================")
	lines.append(f"Arquitectura: 3 -> {hidden_layers} -> {len(labels)}")
	lines.append(f"Epocas: {epochs}")
	lines.append(f"Accuracy entrenamiento: {train_accuracy:.4f}")
	lines.append(f"Accuracy prueba (20%): {test_accuracy:.4f}")
	lines.append("")
	lines.append("Mapeo de etiquetas a costos del robot:")
	for label in labels:
		lines.append(f"- {label} -> costo {terrain_cost_map[label]}")

	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	args = _build_arg_parser().parse_args()

	x, y, labels, _ = load_color_dataset(args.dataset)
	x_train, x_test, y_train, y_test = split_train_test(x, y, test_size=0.2, seed=args.seed)

	terrain_cost_map = build_terrain_cost_map(labels)

	mlp = MLPClassifierNumpy(
		input_size=3,
		hidden_sizes=tuple(args.hidden_layers),
		output_size=len(labels),
		seed=args.seed,
	)

	history = mlp.fit_sgd(
		x_train=x_train,
		y_train=y_train,
		x_test=x_test,
		y_test=y_test,
		epochs=args.epochs,
		learning_rate=args.learning_rate,
		batch_size=args.batch_size,
		seed=args.seed,
	)

	final_metrics = history[-1]
	print("\n=== Task 2.1 (MLP numpy) ===")
	print(f"Muestras entrenamiento: {x_train.shape[0]}")
	print(f"Muestras prueba: {x_test.shape[0]}")
	print(f"Clases: {labels}")
	print(f"Accuracy entrenamiento: {final_metrics.train_accuracy:.4f}")
	print(f"Accuracy prueba (20%): {final_metrics.test_accuracy:.4f}")

	model_path = DEFAULT_OUTPUT_DIR / "color_mlp_weights.npz"
	mlp.save(str(model_path))
	labels_path = DEFAULT_OUTPUT_DIR / "color_mlp_labels.txt"
	labels_path.write_text("\n".join(labels), encoding="utf-8")

	report_path = DEFAULT_OUTPUT_DIR / "task2_1_report.txt"
	_write_report(
		report_path=report_path,
		labels=labels,
		terrain_cost_map=terrain_cost_map,
		test_accuracy=final_metrics.test_accuracy,
		train_accuracy=final_metrics.train_accuracy,
		epochs=args.epochs,
		hidden_layers=tuple(args.hidden_layers),
	)

	print(f"Pesos del modelo guardados en: {model_path}")
	print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
	np.set_printoptions(precision=4, suppress=True)
	main()
