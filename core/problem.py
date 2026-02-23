"""Model of the generic and specific problem for the maze."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Optional, Sequence, Tuple, TypeVar

from maze_io.discretize import CellType, GridRepresentation

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")

class SearchProblemInterface(ABC, Generic[StateType, ActionType]):
    """Mandatory generic template to define any AI search problem."""
    
    def __init__(self, initial_state: StateType):
        self.initial_state = initial_state

    @abstractmethod
    def is_goal_state(self, state_to_evaluate: StateType) -> bool:
        """Asks if the place where we are standing is the finish line."""
        pass

    @abstractmethod
    def possible_actions(self, current_state: StateType) -> Iterable[ActionType]:
        """Asks which movements (up, down, left, right) can be done from here."""
        pass

    @abstractmethod
    def resulting_state(self, current_state: StateType, action_taken: ActionType) -> StateType:
        """Tells us where we will end up if we make a specific movement from this place."""
        pass

    @abstractmethod
    def step_cost(self, current_state: StateType, action_taken: ActionType, next_state: StateType) -> float:
        """Indicates how much 'energy' or 'cost' we will spend when taking this step."""
        pass

    def remaining_distance_estimation(self, current_state: StateType) -> float:
        """A heuristic to know how far we are from the goal 'as the crow flies'. By default, returns 0."""
        return 0.0


Coordinate = Tuple[int, int]
Movement = Tuple[int, int]
POSSIBLE_NEIGHBOR_MOVEMENTS: Sequence[Movement] = ((-1, 0), (1, 0), (0, -1), (0, 1))

class MazeProblem(SearchProblemInterface[Coordinate, Movement]):
    """Translator of a map (image) to the language understood by the algorithms."""
    
    def __init__(self, maze_representation: GridRepresentation, cost_fn: Optional[Callable[[Coordinate], float]] = None):
        super().__init__(initial_state=maze_representation.start)
        
        self.gridded_map = maze_representation.grid
        self.winning_exit_cells = maze_representation.goals
        self.cost_fn = cost_fn
        
        if len(self.winning_exit_cells) == 0:
            raise ValueError("It's impossible to play this maze! There is no visible goal or exit door.")
            
        self.safe_box_of_goals = set(self.winning_exit_cells)
        self.map_height, self.map_width = self.gridded_map.shape

    def is_goal_state(self, state_to_evaluate: Coordinate) -> bool:
        return state_to_evaluate in self.safe_box_of_goals

    def possible_actions(self, current_state: Coordinate) -> Iterable[Movement]:
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
        if self.cost_fn is not None:
            return self.cost_fn(next_state)
        return 1.0

    def remaining_distance_estimation(self, current_state: Coordinate) -> float:
        current_row, current_column = current_state
        possible_distances_to_goals = []
        
        for goal_row, goal_column in self.winning_exit_cells:
            estimated_distance = abs(current_row - goal_row) + abs(current_column - goal_column)
            possible_distances_to_goals.append(estimated_distance)
            
        if possible_distances_to_goals:
            return min(possible_distances_to_goals)
        else:
            return 0.0

    def _am_i_inside_the_map(self, future_row: int, future_column: int) -> bool:
        am_i_inside_top_and_bottom = 0 <= future_row < self.map_height
        am_i_inside_left_and_right = 0 <= future_column < self.map_width
        return am_i_inside_top_and_bottom and am_i_inside_left_and_right

    def _is_a_path_where_i_can_walk(self, future_row: int, future_column: int) -> bool:
        mathematical_type_of_cell = int(self.gridded_map[future_row, future_column])
        is_free_or_white_cell = mathematical_type_of_cell == CellType.FREE
        is_our_home_cell = mathematical_type_of_cell == CellType.START
        is_final_relief_cell = mathematical_type_of_cell == CellType.GOAL
        
        return is_free_or_white_cell or is_our_home_cell or is_final_relief_cell

def build_problem_from_maze(maze: GridRepresentation, cost_fn: Optional[Callable[[Coordinate], float]] = None) -> MazeProblem:
    return MazeProblem(maze, cost_fn=cost_fn)
