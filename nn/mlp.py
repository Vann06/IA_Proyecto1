"""Red Neuronal Multicapa (MLP) implementada desde cero con NumPy."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Convierte salidas brutas en probabilidades por clase (suma = 1)."""
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
    """Red Neuronal Multicapa hecha con NumPy: Input(RGB) → Capas Ocultas → Output(clase de terreno)."""

    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], output_size: int, seed: int = 42) -> None:
        if input_size <= 0 or output_size <= 1:
            raise ValueError("input_size debe ser > 0 y output_size debe ser > 1")
        if any(s <= 0 for s in hidden_sizes):
            raise ValueError("Todas las capas ocultas deben tener tamaño > 0")

        layer_sizes = [input_size, *hidden_sizes, output_size]
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        # Inicialización He: std = sqrt(2/fan_in), óptima para redes con ReLU
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            he_std = np.sqrt(2.0 / in_size)
            self.weights.append(rng.normal(0.0, he_std, size=(in_size, out_size)).astype(np.float64))
            self.biases.append(np.zeros((1, out_size), dtype=np.float64))

    def _forward(self, x_batch: np.ndarray) -> tuple[list, list]:
        """Forward Pass: propaga los datos por cada capa aplicando ReLU (ocultas) y Softmax (salida)."""
        activations = [x_batch]
        pre_activations: list[np.ndarray] = []
        current = x_batch
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            current = _relu(z)
            activations.append(current)
        z_out = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z_out)
        activations.append(_softmax(z_out))
        return activations, pre_activations

    def _compute_loss(self, y_probs: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-Entropy Categórica: mide qué tan equivocadas son las predicciones."""
        epsilon = 1e-12
        clipped = np.clip(y_probs, epsilon, 1.0 - epsilon)
        return float(-np.mean(np.log(clipped[np.arange(y_true.shape[0]), y_true])))

    def _backward(self, activations: list, pre_activations: list, y_true: np.ndarray) -> tuple[list, list]:
        """Backpropagation: calcula el gradiente del error respecto a cada peso (regla de la cadena)."""
        batch_size = y_true.shape[0]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        delta = activations[-1].copy()
        delta[np.arange(batch_size), y_true] -= 1.0
        delta /= batch_size
        grad_w[-1] = activations[-2].T @ delta
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * _relu_derivative(pre_activations[i])
            grad_w[i] = activations[i].T @ delta
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)
        return grad_w, grad_b

    def _update_parameters(self, grad_w: list, grad_b: list, learning_rate: float) -> None:
        """Descenso de Gradiente: ajusta los pesos en la dirección que reduce el error."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i]  -= learning_rate * grad_b[i]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0][-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Retorna el índice de la clase con mayor probabilidad (usado por el live_cost_predictor de A*)."""
        return np.argmax(self.predict_proba(x), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def fit_sgd(self, x_train, y_train, x_test, y_test,
                epochs=100, learning_rate=0.05, batch_size=32, seed=42) -> list[EpochMetrics]:
        """Entrena la red con SGD en mini-batches: Forward → Loss → Backward → Update, por cada época."""
        if epochs <= 0: raise ValueError("epochs debe ser > 0")
        if learning_rate <= 0: raise ValueError("learning_rate debe ser > 0")
        if batch_size <= 0: raise ValueError("batch_size debe ser > 0")

        rng = np.random.default_rng(seed)
        metrics_history: list[EpochMetrics] = []
        samples = x_train.shape[0]

        for epoch in range(1, epochs + 1):
            idx = np.arange(samples)
            rng.shuffle(idx)
            x_epoch, y_epoch = x_train[idx], y_train[idx]
            epoch_losses: list[float] = []
            for start in range(0, samples, batch_size):
                end = min(start + batch_size, samples)
                x_batch, y_batch = x_epoch[start:end], y_epoch[start:end]
                activations, pre_activations = self._forward(x_batch)
                epoch_losses.append(self._compute_loss(activations[-1], y_batch))
                grad_w, grad_b = self._backward(activations, pre_activations, y_batch)
                self._update_parameters(grad_w, grad_b, learning_rate)
            metrics_history.append(EpochMetrics(
                epoch=epoch,
                train_loss=float(np.mean(epoch_losses)),
                train_accuracy=self.accuracy(x_train, y_train),
                test_accuracy=self.accuracy(x_test, y_test),
            ))
        return metrics_history

    def save(self, path: str) -> None:
        """Guarda los pesos y sesgos en un archivo .npz para no tener que volver a entrenar."""
        payload = {f"w{i}": w for i, w in enumerate(self.weights)}
        payload.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, **payload)
