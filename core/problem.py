"""Modelo del problema del laberinto a partir del grid discreto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from maze_io.discretize import CellType, GridRepresentation

Coordinate = Tuple[int, int]
_NEIGHBORS: Sequence[Coordinate] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class MazeProblem:
	grid: np.ndarray
	start: Coordinate
	goals: Sequence[Coordinate]

	def __post_init__(self) -> None:
		if not len(self.goals):
			raise ValueError("MazeProblem requiere al menos un objetivo")
		self._goal_set = set(self.goals)
		self.height, self.width = self.grid.shape

	def is_goal(self, state: Coordinate) -> bool:
		return state in self._goal_set

	def neighbors(self, state: Coordinate) -> Iterable[Tuple[Coordinate, int]]:
		row, col = state
		for d_row, d_col in _NEIGHBORS:
			next_row = row + d_row
			next_col = col + d_col
			if self._in_bounds(next_row, next_col) and self._is_walkable(next_row, next_col):
				yield (next_row, next_col), 1  # costo uniforme

	def heuristic(self, state: Coordinate) -> float:
		distances = [abs(state[0] - r) + abs(state[1] - c) for r, c in self.goals]
		return min(distances) if distances else 0.0

	def _in_bounds(self, row: int, col: int) -> bool:
		return 0 <= row < self.height and 0 <= col < self.width

	def _is_walkable(self, row: int, col: int) -> bool:
		cell = int(self.grid[row, col])
		return cell in (CellType.FREE, CellType.START, CellType.GOAL)


def build_problem_from_grid(grid_representation: GridRepresentation) -> MazeProblem:
	return MazeProblem(
		grid=grid_representation.grid,
		start=grid_representation.start,
		goals=grid_representation.goals,
	)