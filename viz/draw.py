"""Rutinas de visualización para el laberinto discreto.

Se reutiliza la imagen original para mostrar la discretización con una grilla,
y se generan también visualizaciones puramente discretas (basadas en la matriz
del grid) para el camino encontrado por la búsqueda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from maze_io.discretize import CellType, GridRepresentation


Coordinate = Tuple[int, int]


def save_discretization_overlay(
	image: np.ndarray,
	grid: GridRepresentation,
	output_path: Path | str,
	show: bool = False,
) -> None:
	"""Dibuja la grilla discreta sobre la *imagen original*.

	- Las paredes se dejan tal cual en la imagen (negro).
	- El inicio (START) se marca con un recuadro rojo semitransparente.
	- Los objetivos (GOAL) se marcan con recuadros verdes semitransparentes.
	- Todas las celdas muestran líneas de la grilla sobre el fondo.
	"""

	output_path = Path(output_path)

	rows, cols = grid.grid.shape
	tile_size = grid.tile_size

	fig, ax = plt.subplots()
	ax.imshow(image.astype(np.uint8))

	for row in range(rows):
		for col in range(cols):
			cell = int(grid.grid[row, col])

			# Coordenadas del tile en píxeles
			x0 = col * tile_size
			y0 = row * tile_size

			facecolor = "none"
			alpha = 0.0

			if cell == CellType.START:
				facecolor = "red"
				alpha = 0.4
			elif cell == CellType.GOAL:
				facecolor = "green"
				alpha = 0.4

			# Se dibuja siempre el borde de la celda para que la grilla sea visible
			rect = Rectangle(
				(x0, y0),
				tile_size,
				tile_size,
				linewidth=0.5,
				edgecolor="white",
				facecolor=facecolor,
				alpha=alpha,
			)
			ax.add_patch(rect)

	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_axis_off()

	fig.tight_layout(pad=0)
	fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

	if show:
		plt.show()
	plt.close(fig)


def save_path_on_grid(
	grid: GridRepresentation,
	path: Iterable[Coordinate],
	output_path: Path | str,
	show: bool = True,
	draw_path: bool = False,
) -> None:
	"""Visualiza la discretización y, opcionalmente, el camino.

	Basado únicamente en la *matriz discreta* (no la imagen original).

	- Paredes  -> negro
	- Libres   -> blanco
	- Inicio   -> rojo
	- Objetivo -> verde
	- Camino   -> azul (solo si ``draw_path=True``)
	"""

	output_path = Path(output_path)

	grid_array = grid.grid
	rows, cols = grid_array.shape

	# Imagen pequeña (rows x cols) que luego se escala con interpolation='nearest'
	vis = np.zeros((rows, cols, 3), dtype=np.uint8)

	# Colores base por tipo de celda
	base_colors = {
		CellType.WALL: np.array([0, 0, 0], dtype=np.uint8),       # negro
		CellType.FREE: np.array([255, 255, 255], dtype=np.uint8), # blanco
		CellType.START: np.array([255, 0, 0], dtype=np.uint8),    # rojo
		CellType.GOAL: np.array([0, 255, 0], dtype=np.uint8),     # verde
	}

	for r in range(rows):
		for c in range(cols):
			cell = int(grid_array[r, c])
			vis[r, c] = base_colors.get(cell, np.array([255, 255, 255], dtype=np.uint8))

	# Pintar el camino encima (salvo inicio y objetivos), si se solicita
	if draw_path:
		start = grid.start
		goal_set = set(grid.goals)
		path_color = np.array([0, 0, 255], dtype=np.uint8)  # azul

		for (r, c) in path:
			if (r, c) == start or (r, c) in goal_set:
				continue
			vis[r, c] = path_color

	fig, ax = plt.subplots()
	ax.imshow(vis, interpolation="nearest", origin="upper")

	# Dibujar la grilla sobre la visualización discreta (en blanco)
	ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
	ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
	ax.grid(which="minor", color="white", linewidth=0.5)

	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_axis_off()

	fig.tight_layout(pad=0)
	fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

	if show:
		plt.show()
	plt.close(fig)
