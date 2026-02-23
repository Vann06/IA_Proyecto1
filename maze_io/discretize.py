"""Herramientas para transformar una imagen RGB en un grid discreto."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

import numpy as np


class CellType:
    WALL  = 0  # Negro: intransitable
    FREE  = 1  # Blanco (u otro color): pasillo libre
    START = 2  # Rojo: posición de inicio
    GOAL  = 3  # Verde: posición de meta


@dataclass
class GridRepresentation:
    """Resultado de la discretización: cuadrícula, inicio, metas y tamaño de tile."""
    grid: np.ndarray
    start: Tuple[int, int]
    goals: List[Tuple[int, int]]
    tile_size: int


def discretize_image(
    image: np.ndarray,
    tile_size: int = 10,
    tolerance: float = 45.0,
    is_complex: bool = False,
) -> GridRepresentation:
    """Convierte la imagen en una cuadrícula agrupando píxeles en bloques de tile_size×tile_size.
    
    Cada bloque se clasifica por su color promedio: WALL, FREE, START o GOAL.
    El modo 'is_complex' usa distancia euclidiana para imágenes de alta precisión.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")
    if tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    
    image = image.astype(np.float32)
    height, width, _ = image.shape
    rows = ceil(height / tile_size)
    cols = ceil(width / tile_size)
    grid = np.full((rows, cols), CellType.WALL, dtype=np.uint8)
    start = None
    start_candidates: List[Tuple[int, int]] = []
    goals: List[Tuple[int, int]] = []

    colors = {
        CellType.FREE:  np.array([255, 255, 255], dtype=np.float32),
        CellType.WALL:  np.array([0,   0,   0  ], dtype=np.float32),
        CellType.START: np.array([255, 0,   0  ], dtype=np.float32),
        CellType.GOAL:  np.array([0,   255, 0  ], dtype=np.float32),
    }

    for row in range(rows):
        y0 = row * tile_size
        y1 = min(y0 + tile_size, height)
        for col in range(cols):
            x0 = col * tile_size
            x1 = min(x0 + tile_size, width)
            tile = image[y0:y1, x0:x1]
            avg_color = tile.reshape(-1, 3).mean(axis=0)

            if is_complex:
                cell_type = _classify_color_euclidean(avg_color, colors, tolerance)
                # Si el tile parece libre pero contiene píxeles oscuros, es pared
                if cell_type == CellType.FREE and tile_size > 1:
                    min_color = tile.reshape(-1, 3).min(axis=0)
                    if _classify_color_euclidean(min_color, colors, tolerance) == CellType.WALL:
                        cell_type = CellType.WALL
            else:
                cell_type = _classify_color(avg_color, colors, tolerance)

            grid[row, col] = cell_type

            if cell_type == CellType.START:
                start_candidates.append((row, col))
            elif cell_type == CellType.GOAL:
                goals.append((row, col))

    # Si hay múltiples tiles rojos, elegimos el del centro de la región
    if start_candidates:
        rows_idx = np.array([r for r, _ in start_candidates], dtype=np.float32)
        cols_idx = np.array([c for _, c in start_candidates], dtype=np.float32)
        mean_r, mean_c = float(rows_idx.mean()), float(cols_idx.mean())
        dists = [abs(r - mean_r) + abs(c - mean_c) for r, c in start_candidates]
        start = start_candidates[int(np.argmin(dists))]
    else:
        # Fallback: primera celda libre del mapa
        start = next(((r, c) for r in range(rows) for c in range(cols) if grid[r, c] == CellType.FREE), (0, 0))

    if not goals:
        # Fallback: última celda libre distinta al inicio
        goal_candidate = next(
            ((r, c) for r in range(rows - 1, -1, -1) for c in range(cols - 1, -1, -1)
             if grid[r, c] == CellType.FREE and (r, c) != start), start
        )
        goals = [goal_candidate]

    return GridRepresentation(grid=grid, start=start, goals=goals, tile_size=tile_size)


def _classify_color(color: np.ndarray, reference_colors: dict, tolerance: float) -> int:
    """Clasifica el color usando dominancia de canal RGB.
    
    Oscuro → WALL | R domina → START | G domina → GOAL | otro → FREE.
    """
    color = color.astype(np.float32)
    r, g, b = color
    if (r + g + b) / 3.0 < 60.0:
        return CellType.WALL
    if (r - max(g, b)) > tolerance:
        return CellType.START
    if (g - max(r, b)) > tolerance:
        return CellType.GOAL
    return CellType.FREE


def _classify_color_euclidean(color: np.ndarray, reference_colors: dict, tolerance: float) -> int:
    """Clasifica por distancia euclidiana al color de referencia más cercano (para imágenes complejas)."""
    distances = {label: np.linalg.norm(color - ref) for label, ref in reference_colors.items()}
    label = min(distances, key=distances.get)
    return label if distances[label] <= tolerance else CellType.FREE
