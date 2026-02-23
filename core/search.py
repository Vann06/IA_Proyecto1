"""Search algorithms explicitly described as the agent's mental processes."""

# El truco de este módulo: un solo algoritmo maestro que cambia su comportamiento
# dependiendo de la estructura de datos que se le inyecta (BFS=Cola, DFS=Pila, A*=Heap).

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from core.problem import SearchProblemInterface, StateType

@dataclass
class FinalSearchReport:
    """Resultado final del algoritmo: éxito, camino, costo y nodos explorados."""
    success_reaching_goal: bool
    history_of_winning_steps: List[Any]
    wasted_energy_cost: float
    amount_of_checked_boxes_before_winning: int

class BreadcrumbToRememberPath:
    """Nodo del árbol de búsqueda. Guarda su posición, su padre y el costo acumulado g(n)."""
    def __init__(self, current_location: StateType, previous_parent_breadcrumb: Optional['BreadcrumbToRememberPath'] = None, step_that_brought_us_here: Any = None, accumulated_used_fuel: float = 0.0):
        self.current_location = current_location
        self.previous_parent_breadcrumb = previous_parent_breadcrumb
        self.step_that_brought_us_here = step_that_brought_us_here
        self.accumulated_used_fuel = accumulated_used_fuel

    def follow_breadcrumbs_back_home(self) -> List[StateType]:
        """Reconstruye el camino ganador siguiendo la cadena de punteros padre → inicio."""
        breadcrumb_i_am_looking_at_now = self
        steps_in_retrospect = []
        while breadcrumb_i_am_looking_at_now is not None:
            steps_in_retrospect.append(breadcrumb_i_am_looking_at_now.current_location)
            breadcrumb_i_am_looking_at_now = breadcrumb_i_am_looking_at_now.previous_parent_breadcrumb
        return list(reversed(steps_in_retrospect))


def master_graph_search_algorithm(
    mathematical_problem: SearchProblemInterface, 
    border_rest_zone: Any, 
    decide_who_advances_now: Callable, 
    enqueue_next_discovery: Callable
) -> FinalSearchReport:
    """Algoritmo genérico de búsqueda en grafos.
    
    BFS, DFS y A* comparten este motor; lo que cambia es quién sale primero de la frontera.
    Un 'cuaderno' de visitados evita explorar el mismo nodo dos veces.
    """
    original_starting_breadcrumb = BreadcrumbToRememberPath(mathematical_problem.initial_state)
    
    if mathematical_problem.is_goal_state(original_starting_breadcrumb.current_location):
        immediate_history = original_starting_breadcrumb.follow_breadcrumbs_back_home()
        return FinalSearchReport(True, immediate_history, original_starting_breadcrumb.accumulated_used_fuel, 0)
    
    enqueue_next_discovery(border_rest_zone, original_starting_breadcrumb)
    
    notebook_of_places_i_already_stepped_on: Set[StateType] = set()
    notebook_of_visualized_places_waiting_their_turn: Set[StateType] = {original_starting_breadcrumb.current_location}
    number_of_times_i_took_a_step = 0

    while border_rest_zone:
        current_breadcrumb_in_my_hand = decide_who_advances_now(border_rest_zone)
        position = current_breadcrumb_in_my_hand.current_location
        notebook_of_visualized_places_waiting_their_turn.discard(position)
        notebook_of_places_i_already_stepped_on.add(position)
        number_of_times_i_took_a_step += 1
        
        if mathematical_problem.is_goal_state(position):
            return FinalSearchReport(
                success_reaching_goal=True, 
                history_of_winning_steps=current_breadcrumb_in_my_hand.follow_breadcrumbs_back_home(), 
                wasted_energy_cost=current_breadcrumb_in_my_hand.accumulated_used_fuel, 
                amount_of_checked_boxes_before_winning=number_of_times_i_took_a_step
            )
            
        for door in mathematical_problem.possible_actions(position):
            place_where_i_end_up = mathematical_problem.resulting_state(position, door)
            effort = mathematical_problem.step_cost(position, door, place_where_i_end_up)
            updated_tiredness_level = current_breadcrumb_in_my_hand.accumulated_used_fuel + effort
            newly_created_breadcrumb = BreadcrumbToRememberPath(place_where_i_end_up, current_breadcrumb_in_my_hand, door, updated_tiredness_level)
            
            if place_where_i_end_up not in notebook_of_places_i_already_stepped_on and \
               place_where_i_end_up not in notebook_of_visualized_places_waiting_their_turn:
                enqueue_next_discovery(border_rest_zone, newly_created_breadcrumb)
                notebook_of_visualized_places_waiting_their_turn.add(place_where_i_end_up)

    return FinalSearchReport(False, [], 0.0, number_of_times_i_took_a_step)


def use_relaxed_and_egalitarian_search_bfs(problem: SearchProblemInterface) -> FinalSearchReport:
    """BFS: Cola FIFO → explora por capas, garantiza el camino con menos pasos."""
    the_long_line_at_the_bank = deque()
    return master_graph_search_algorithm(
        mathematical_problem=problem, 
        border_rest_zone=the_long_line_at_the_bank, 
        decide_who_advances_now=lambda q: q.popleft(), 
        enqueue_next_discovery=lambda q, n: q.append(n)
    )

def use_obsessive_but_fast_search_dfs(problem: SearchProblemInterface) -> FinalSearchReport:
    """DFS: Pila LIFO → explora caminos completos antes de retroceder, muy rápido pero no óptimo."""
    stack_of_papers_on_my_desk = []
    return master_graph_search_algorithm(
        mathematical_problem=problem, 
        border_rest_zone=stack_of_papers_on_my_desk, 
        decide_who_advances_now=lambda s: s.pop(), 
        enqueue_next_discovery=lambda s, n: s.append(n)
    )

def use_artificial_intelligence_type_a_star(maze_problem: SearchProblemInterface) -> FinalSearchReport:
    """A*: prioriza nodos con menor f(n) = g(n) + h(n).
    
    g(n) = costo real acumulado (puede ser dinámico vía Red Neuronal).
    h(n) = heurística Manhattan hacia la meta más cercana.
    Actualiza el costo de un nodo si encuentra un camino más barato.
    """
    native_position = maze_problem.initial_state
    line_that_favors_important_people = []
    a_simple_tiebreaker_clock = itertools.count()
    h0 = maze_problem.remaining_distance_estimation(native_position)
    heapq.heappush(line_that_favors_important_people, (h0, next(a_simple_tiebreaker_clock), 0.0, native_position))

    # Hilo de Ariadna: recuerda el padre de cada nodo para reconstruir el camino
    ariadnes_thread_where_i_come_from: Dict[StateType, Optional[StateType]] = {native_position: None}
    purse_of_accumulated_real_cost: Dict[StateType, float] = {native_position: 0.0}
    general_observation_count = 0

    while line_that_favors_important_people:
        _, _, burned_real_money, cell_under_the_magnifying_glass_right_now = heapq.heappop(line_that_favors_important_people)
        general_observation_count += 1

        if maze_problem.is_goal_state(cell_under_the_magnifying_glass_right_now):
            route_painted_on_the_map = []
            cell_in_reverse = cell_under_the_magnifying_glass_right_now
            while cell_in_reverse is not None:
                route_painted_on_the_map.append(cell_in_reverse)
                cell_in_reverse = ariadnes_thread_where_i_come_from[cell_in_reverse]
            route_painted_on_the_map.reverse()
            return FinalSearchReport(True, route_painted_on_the_map, burned_real_money, general_observation_count)

        for new_path in maze_problem.possible_actions(cell_under_the_magnifying_glass_right_now):
            cell_where_we_will_arrive = maze_problem.resulting_state(cell_under_the_magnifying_glass_right_now, new_path)
            # step_cost llama al predictor de la Red Neuronal si está activo
            cost_of_this_transition = maze_problem.step_cost(cell_under_the_magnifying_glass_right_now, new_path, cell_where_we_will_arrive)
            new_g = burned_real_money + cost_of_this_transition
            
            is_new = cell_where_we_will_arrive not in purse_of_accumulated_real_cost
            is_cheaper = not is_new and new_g < purse_of_accumulated_real_cost[cell_where_we_will_arrive]

            if is_new or is_cheaper:
                purse_of_accumulated_real_cost[cell_where_we_will_arrive] = new_g
                f = new_g + maze_problem.remaining_distance_estimation(cell_where_we_will_arrive)
                heapq.heappush(line_that_favors_important_people, (f, next(a_simple_tiebreaker_clock), new_g, cell_where_we_will_arrive))
                ariadnes_thread_where_i_come_from[cell_where_we_will_arrive] = cell_under_the_magnifying_glass_right_now

    return FinalSearchReport(False, [], -1.0, general_observation_count)
