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
                if start is not None:
                    raise ValueError("Multiple starting positions detected in the image")
                start = (row, col)
            elif cell_type == CellType.GOAL:
                goals.append((row, col))

    if start is None:
        raise ValueError("No starting position (red tile) found in the image")
    if not goals:
        raise ValueError("No goal positions (green tiles) found in the image")

    return GridRepresentation(grid=grid, start=start, goals=goals, tile_size=tile_size)

# Clasificar el color promedio de cada tile comparandolo con los
#  colores de referencia y asignando la etiqueta correspondiente
def _classify_color(
    color: np.ndarray,
    reference_colors: dict,
    tolerance: float,
) -> int:
# retorna la etiqueta de la celda (libre, pared, inicio, objetivo) basada 
# en la distancia al color promedio del tile
    distances = {
        label: np.linalg.norm(color - reference)
        for label, reference in reference_colors.items()
    }

    label = min(distances, key=distances.get)
    if distances[label] <= tolerance:
        return label

    return CellType.FREE