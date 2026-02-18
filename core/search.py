"""Algoritmos de búsqueda."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.problem import Coordinate, MazeProblem


@dataclass
class SearchResult:
	success: bool
	path: List[Coordinate]
	cost: int
	explored: int


def a_star_search(problem: MazeProblem) -> SearchResult:
	start = problem.start
	frontier = []
	counter = itertools.count()
	heapq.heappush(frontier, (problem.heuristic(start), next(counter), 0, start))

	came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
	g_costs: Dict[Coordinate, int] = {start: 0}
	explored = 0

	while frontier:
		_, _, current_cost, current = heapq.heappop(frontier)
		explored += 1

		if problem.is_goal(current):
			path = _reconstruct_path(came_from, current)
			return SearchResult(True, path, current_cost, explored)

		for neighbor, step_cost in problem.neighbors(current):
			new_cost = current_cost + step_cost
			if neighbor not in g_costs or new_cost < g_costs[neighbor]:
				g_costs[neighbor] = new_cost
				priority = new_cost + problem.heuristic(neighbor)
				heapq.heappush(frontier, (priority, next(counter), new_cost, neighbor))
				came_from[neighbor] = current

	return SearchResult(False, [], -1, explored)


def _reconstruct_path(
	came_from: Dict[Coordinate, Optional[Coordinate]],
	current: Coordinate,
) -> List[Coordinate]:
	path: List[Coordinate] = [current]
	while came_from[current] is not None:
		current = came_from[current]
		path.append(current)
	path.reverse()
	return path