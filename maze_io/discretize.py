"""Herramientas para transformar una imagen RGB en un grid discreto."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

import numpy as np


class CellType:
	WALL = 0
	FREE = 1
	START = 2
	GOAL = 3


@dataclass
class GridRepresentation:
	grid: np.ndarray
	start: Tuple[int, int]
	goals: List[Tuple[int, int]]
	tile_size: int


def discretize_image(
	image: np.ndarray,
	tile_size: int = 10,
	tolerance: float = 45.0,
) -> GridRepresentation:
	# Convertir la imagen RGB a un grid discreto de celdas (libre, pared, inicio, objetivo)
	if image.ndim != 3 or image.shape[2] != 3:
		raise ValueError("Expected an RGB image with shape (H, W, 3)")
	if tile_size <= 0:
		raise ValueError("tile_size must be a positive integer")
	# Definicion de variables dentro del entorno 
	image = image.astype(np.float32)
	height, width, _ = image.shape
	rows = ceil(height / tile_size)
	cols = ceil(width / tile_size)
	grid = np.full((rows, cols), CellType.WALL, dtype=np.uint8)
	start = None
	start_candidates: List[Tuple[int, int]] = []
	goals: List[Tuple[int, int]] = []

	# Los colores de referencia para clasificar cada celda
	colors = {
		CellType.FREE: np.array([255, 255, 255], dtype=np.float32),
		CellType.WALL: np.array([0, 0, 0], dtype=np.float32),
		CellType.START: np.array([255, 0, 0], dtype=np.float32),
		CellType.GOAL: np.array([0, 255, 0], dtype=np.float32),
	}

	# Iterar sobre cada tile, calcular su color promedio y clasificarlo
	for row in range(rows):
		y0 = row * tile_size
		y1 = min(y0 + tile_size, height)
		for col in range(cols):
			x0 = col * tile_size
			x1 = min(x0 + tile_size, width)
			tile = image[y0:y1, x0:x1]
			avg_color = tile.reshape(-1, 3).mean(axis=0)

			cell_type = _classify_color(avg_color, colors, tolerance)
			grid[row, col] = cell_type

			if cell_type == CellType.START:
				start_candidates.append((row, col))
			elif cell_type == CellType.GOAL:
				goals.append((row, col))

	# Elegir la posición de inicio
	if start_candidates:
		# Si hay varios tiles rojos (una región de inicio), elegir uno "representativo".
		# Aquí usamos el centro aproximado de la región calculando el promedio
		# de filas y columnas y tomando el candidato más cercano.
		rows_idx = np.array([r for r, _ in start_candidates], dtype=np.float32)
		cols_idx = np.array([c for _, c in start_candidates], dtype=np.float32)
		mean_r = float(rows_idx.mean())
		mean_c = float(cols_idx.mean())
		dists = [abs(r - mean_r) + abs(c - mean_c) for r, c in start_candidates]
		best_idx = int(np.argmin(dists))
		start = start_candidates[best_idx]
	else:
		# Si no se encontró inicio explícito, elegir valores por defecto
		start_candidate = None
		for r in range(rows):
			for c in range(cols):
				if grid[r, c] == CellType.FREE:
					start_candidate = (r, c)
					break
			if start_candidate is not None:
				break
		# Si por alguna razón no hay celdas libres, usar (0, 0)
		start = start_candidate or (0, 0)

	if not goals:
		# Elegir alguna celda libre como objetivo (distinta del inicio si es posible)
		goal_candidate = None
		for r in range(rows - 1, -1, -1):
			for c in range(cols - 1, -1, -1):
				if grid[r, c] == CellType.FREE and (r, c) != start:
					goal_candidate = (r, c)
					break
			if goal_candidate is not None:
				break
		if goal_candidate is None:
			goal_candidate = start
		goals = [goal_candidate]

	return GridRepresentation(grid=grid, start=start, goals=goals, tile_size=tile_size)


def _classify_color(
	color: np.ndarray,
	reference_colors: dict,
	tolerance: float,
) -> int:
	"""Clasifica un color promedio en WALL/FREE/START/GOAL.

	En lugar de exigir exactamente (255, 0, 0) o (0, 255, 0), se detectan
	"rojos" y "verdes" dominantes permitiendo variaciones.
	
	- Si el color es muy oscuro => WALL.
	- Si R domina claramente sobre G y B => START.
	- Si G domina claramente sobre R y B => GOAL.
	- En otro caso => FREE.
	"""

	# Asegurar tipo float
	color = color.astype(np.float32)
	r, g, b = color
	brightness = (r + g + b) / 3.0

	# Umbral para considerar una celda como pared (oscura)
	dark_threshold = 60.0
	if brightness < dark_threshold:
		return CellType.WALL

	# Usamos el parámetro tolerance como diferencia mínima de canal dominante
	delta = tolerance

	max_gb = max(g, b)
	max_rb = max(r, b)

	# Dominancia de rojo
	if (r - max_gb) > delta:
		return CellType.START

	# Dominancia de verde
	if (g - max_rb) > delta:
		return CellType.GOAL

	# Si no es ni muy oscuro ni claramente rojo/verde, lo tratamos como libre
	return CellType.FREE
