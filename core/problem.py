"""Model of the generic and specific problem for the maze."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Optional, Sequence, Tuple, TypeVar

from maze_io.discretize import CellType, GridRepresentation

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")

# Interfaz genérica del problema de búsqueda – cualquier algoritmo puede usarla sin cambios
class SearchProblemInterface(ABC, Generic[StateType, ActionType]):
    """Mandatory generic template to define any AI search problem."""
    
    def __init__(self, initial_state: StateType):
        self.initial_state = initial_state

    @abstractmethod
    def is_goal_state(self, state_to_evaluate: StateType) -> bool:
        """¿Estamos parados en la meta?"""
        pass

    @abstractmethod
    def possible_actions(self, current_state: StateType) -> Iterable[ActionType]:
        """¿Qué movimientos son válidos desde aquí?"""
        pass

    @abstractmethod
    def resulting_state(self, current_state: StateType, action_taken: ActionType) -> StateType:
        """¿A dónde llegamos si hacemos este movimiento?"""
        pass

    @abstractmethod
    def step_cost(self, current_state: StateType, action_taken: ActionType, next_state: StateType) -> float:
        """¿Cuánto cuesta dar este paso?"""
        pass

    def remaining_distance_estimation(self, current_state: StateType) -> float:
        """Heurística: estimación 'a vuelo de pájaro' de cuánto falta para la meta."""
        return 0.0


Coordinate = Tuple[int, int]
Movement = Tuple[int, int]

# Los cuatro movimientos posibles: arriba, abajo, izquierda, derecha (sin diagonal)
POSSIBLE_NEIGHBOR_MOVEMENTS: Sequence[Movement] = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Implementación concreta del problema para nuestro laberinto discretizado
class MazeProblem(SearchProblemInterface[Coordinate, Movement]):
    """Traduce la cuadrícula del laberinto al lenguaje que entienden los algoritmos."""
    
    def __init__(self, maze_representation: GridRepresentation, cost_fn: Optional[Callable[[Coordinate], float]] = None):
        super().__init__(initial_state=maze_representation.start)
        self.gridded_map = maze_representation.grid
        self.winning_exit_cells = maze_representation.goals
        # cost_fn opcional: si existe, A* usará la Red Neuronal para costos dinámicos
        self.cost_fn = cost_fn
        
        if len(self.winning_exit_cells) == 0:
            raise ValueError("It's impossible to play this maze! There is no visible goal or exit door.")
            
        self.safe_box_of_goals = set(self.winning_exit_cells)
        self.map_height, self.map_width = self.gridded_map.shape

    def is_goal_state(self, state_to_evaluate: Coordinate) -> bool:
        return state_to_evaluate in self.safe_box_of_goals

    def possible_actions(self, current_state: Coordinate) -> Iterable[Movement]:
        # Generamos solo los vecinos que existen dentro del mapa y no son paredes
        current_row, current_column = current_state
        for row_jump, column_jump in POSSIBLE_NEIGHBOR_MOVEMENTS:
            future_row = current_row + row_jump
            future_column = current_column + column_jump
            if self._am_i_inside_the_map(future_row, future_column):
                 if self._is_a_path_where_i_can_walk(future_row, future_column):
                    yield (row_jump, column_jump)

    def resulting_state(self, current_state: Coordinate, action_taken: Movement) -> Coordinate:
        current_row, current_column = current_state
        row_jump, column_jump = action_taken
        return (current_row + row_jump, current_column + column_jump)

    def step_cost(self, current_state: Coordinate, action_taken: Movement, next_state: Coordinate) -> float:
        # Si hay Red Neuronal activa, el costo depende del color del terreno; si no, costo fijo 1.0
        if self.cost_fn is not None:
            return self.cost_fn(next_state)
        return 1.0

    def remaining_distance_estimation(self, current_state: Coordinate) -> float:
        # Heurística Manhattan: suma de diferencias en fila y columna hasta la meta más cercana
        current_row, current_column = current_state
        possible_distances_to_goals = [
            abs(current_row - gr) + abs(current_column - gc)
            for gr, gc in self.winning_exit_cells
        ]
        return min(possible_distances_to_goals) if possible_distances_to_goals else 0.0

    def _am_i_inside_the_map(self, future_row: int, future_column: int) -> bool:
        return 0 <= future_row < self.map_height and 0 <= future_column < self.map_width

    def _is_a_path_where_i_can_walk(self, future_row: int, future_column: int) -> bool:
        # El agente puede pisar FREE, START o GOAL; nunca WALL
        cell = int(self.gridded_map[future_row, future_column])
        return cell in (CellType.FREE, CellType.START, CellType.GOAL)

def build_problem_from_maze(maze: GridRepresentation, cost_fn: Optional[Callable[[Coordinate], float]] = None) -> MazeProblem:
    return MazeProblem(maze, cost_fn=cost_fn)
