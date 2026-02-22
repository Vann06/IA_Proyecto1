from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
	return np.maximum(0.0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
	return (x > 0).astype(np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
	shifted = logits - np.max(logits, axis=1, keepdims=True)
	exp_values = np.exp(shifted)
	return exp_values / np.sum(exp_values, axis=1, keepdims=True)


@dataclass
class EpochMetrics:
	epoch: int
	train_loss: float
	train_accuracy: float
	test_accuracy: float


class MLPClassifierNumpy:
	def __init__(
		self,
		input_size: int,
		hidden_sizes: tuple[int, ...],
		output_size: int,
		seed: int = 42,
	) -> None:
		if input_size <= 0 or output_size <= 1:
			raise ValueError("input_size debe ser > 0 y output_size debe ser > 1")
		if any(layer_size <= 0 for layer_size in hidden_sizes):
			raise ValueError("Todas las capas ocultas deben tener tamaño > 0")

		layer_sizes = [input_size, *hidden_sizes, output_size]
		rng = np.random.default_rng(seed)

		self.weights: list[np.ndarray] = []
		self.biases: list[np.ndarray] = []

		for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
			he_std = np.sqrt(2.0 / in_size)
			self.weights.append(rng.normal(0.0, he_std, size=(in_size, out_size)).astype(np.float64))
			self.biases.append(np.zeros((1, out_size), dtype=np.float64))

	def _forward(self, x_batch: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
		activations = [x_batch]
		pre_activations: list[np.ndarray] = []

		current = x_batch
		for layer_index in range(len(self.weights) - 1):
			z = current @ self.weights[layer_index] + self.biases[layer_index]
			pre_activations.append(z)
			current = _relu(z)
			activations.append(current)

		z_out = current @ self.weights[-1] + self.biases[-1]
		pre_activations.append(z_out)
		y_hat = _softmax(z_out)
		activations.append(y_hat)
		return activations, pre_activations

	def _compute_loss(self, y_probs: np.ndarray, y_true: np.ndarray) -> float:
		epsilon = 1e-12
		clipped = np.clip(y_probs, epsilon, 1.0 - epsilon)
		batch_size = y_true.shape[0]
		return float(-np.mean(np.log(clipped[np.arange(batch_size), y_true])))

	def _backward(
		self,
		activations: list[np.ndarray],
		pre_activations: list[np.ndarray],
		y_true: np.ndarray,
	) -> tuple[list[np.ndarray], list[np.ndarray]]:
		batch_size = y_true.shape[0]
		grad_w = [np.zeros_like(w) for w in self.weights]
		grad_b = [np.zeros_like(b) for b in self.biases]

		delta = activations[-1].copy()
		delta[np.arange(batch_size), y_true] -= 1.0
		delta /= batch_size

		grad_w[-1] = activations[-2].T @ delta
		grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

		for layer_index in range(len(self.weights) - 2, -1, -1):
			delta = (delta @ self.weights[layer_index + 1].T) * _relu_derivative(pre_activations[layer_index])
			grad_w[layer_index] = activations[layer_index].T @ delta
			grad_b[layer_index] = np.sum(delta, axis=0, keepdims=True)

		return grad_w, grad_b

	def _update_parameters(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray], learning_rate: float) -> None:
		for index in range(len(self.weights)):
			self.weights[index] -= learning_rate * grad_w[index]
			self.biases[index] -= learning_rate * grad_b[index]

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		activations, _ = self._forward(x)
		return activations[-1]

	def predict(self, x: np.ndarray) -> np.ndarray:
		probabilities = self.predict_proba(x)
		return np.argmax(probabilities, axis=1)

	def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
		predictions = self.predict(x)
		return float(np.mean(predictions == y))

	def fit_sgd(
		self,
		x_train: np.ndarray,
		y_train: np.ndarray,
		x_test: np.ndarray,
		y_test: np.ndarray,
		epochs: int = 100,
		learning_rate: float = 0.05,
		batch_size: int = 32,
		seed: int = 42,
	) -> list[EpochMetrics]:
		if epochs <= 0:
			raise ValueError("epochs debe ser > 0")
		if learning_rate <= 0:
			raise ValueError("learning_rate debe ser > 0")
		if batch_size <= 0:
			raise ValueError("batch_size debe ser > 0")

		rng = np.random.default_rng(seed)
		metrics_history: list[EpochMetrics] = []
		samples = x_train.shape[0]

		for epoch in range(1, epochs + 1):
			shuffled_indices = np.arange(samples)
			rng.shuffle(shuffled_indices)
			x_epoch = x_train[shuffled_indices]
			y_epoch = y_train[shuffled_indices]

			epoch_losses: list[float] = []
			for start in range(0, samples, batch_size):
				end = min(start + batch_size, samples)
				x_batch = x_epoch[start:end]
				y_batch = y_epoch[start:end]

				activations, pre_activations = self._forward(x_batch)
				loss = self._compute_loss(activations[-1], y_batch)
				epoch_losses.append(loss)

				grad_w, grad_b = self._backward(activations, pre_activations, y_batch)
				self._update_parameters(grad_w, grad_b, learning_rate)

			train_loss = float(np.mean(epoch_losses))
			train_accuracy = self.accuracy(x_train, y_train)
			test_accuracy = self.accuracy(x_test, y_test)
			metrics_history.append(
				EpochMetrics(
					epoch=epoch,
					train_loss=train_loss,
					train_accuracy=train_accuracy,
					test_accuracy=test_accuracy,
				)
			)

		return metrics_history

	def save(self, path: str) -> None:
		payload: dict[str, np.ndarray] = {}
		for index, weights in enumerate(self.weights):
			payload[f"w{index}"] = weights
		for index, bias in enumerate(self.biases):
			payload[f"b{index}"] = bias
		np.savez(path, **payload)
