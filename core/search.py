"""Search algorithms explicitly described as the agent's mental processes."""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from core.problem import SearchProblemInterface, StateType

@dataclass
class FinalSearchReport:
    success_reaching_goal: bool
    history_of_winning_steps: List[Any]
    wasted_energy_cost: float
    amount_of_checked_boxes_before_winning: int

class BreadcrumbToRememberPath:
    """A 'breadcrumb' or node in the great tree of possibilities."""
    def __init__(self, current_location: StateType, previous_parent_breadcrumb: Optional['BreadcrumbToRememberPath'] = None, step_that_brought_us_here: Any = None, accumulated_used_fuel: float = 0.0):
        self.current_location = current_location
        self.previous_parent_breadcrumb = previous_parent_breadcrumb
        self.step_that_brought_us_here = step_that_brought_us_here
        self.accumulated_used_fuel = accumulated_used_fuel

    def follow_breadcrumbs_back_home(self) -> List[StateType]:
        """Traces the winning steps back from start to finish."""
        breadcrumb_i_am_looking_at_now = self
        steps_in_retrospect = []
        
        while breadcrumb_i_am_looking_at_now is not None:
            steps_in_retrospect.append(breadcrumb_i_am_looking_at_now.current_location)
            breadcrumb_i_am_looking_at_now = breadcrumb_i_am_looking_at_now.previous_parent_breadcrumb
            
        correctly_ordered_trip_list = list(reversed(steps_in_retrospect))
        return correctly_ordered_trip_list


def master_graph_search_algorithm(
    mathematical_problem: SearchProblemInterface, 
    border_rest_zone: Any, 
    decide_who_advances_now: Callable, 
    enqueue_next_discovery: Callable
) -> FinalSearchReport:
    
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
            triumph_report = FinalSearchReport(
                success_reaching_goal=True, 
                history_of_winning_steps=current_breadcrumb_in_my_hand.follow_breadcrumbs_back_home(), 
                wasted_energy_cost=current_breadcrumb_in_my_hand.accumulated_used_fuel, 
                amount_of_checked_boxes_before_winning=number_of_times_i_took_a_step
            )
            return triumph_report
            
        doors_to_open = mathematical_problem.possible_actions(position)
        
        for door in doors_to_open:
            place_where_i_end_up = mathematical_problem.resulting_state(position, door)
            effort = mathematical_problem.step_cost(position, door, place_where_i_end_up)
            updated_tiredness_level = current_breadcrumb_in_my_hand.accumulated_used_fuel + effort
            
            newly_created_breadcrumb = BreadcrumbToRememberPath(place_where_i_end_up, current_breadcrumb_in_my_hand, door, updated_tiredness_level)
            
            is_a_totally_unknown_place = place_where_i_end_up not in notebook_of_places_i_already_stepped_on
            i_have_not_kept_an_eye_on_it_yet = place_where_i_end_up not in notebook_of_visualized_places_waiting_their_turn
            
            if is_a_totally_unknown_place and i_have_not_kept_an_eye_on_it_yet:
                enqueue_next_discovery(border_rest_zone, newly_created_breadcrumb)
                notebook_of_visualized_places_waiting_their_turn.add(place_where_i_end_up)

    return FinalSearchReport(False, [], 0.0, number_of_times_i_took_a_step)


def use_relaxed_and_egalitarian_search_bfs(problem: SearchProblemInterface) -> FinalSearchReport:
    the_long_line_at_the_bank = deque() # FIFO queue
    return master_graph_search_algorithm(
        mathematical_problem=problem, 
        border_rest_zone=the_long_line_at_the_bank, 
        decide_who_advances_now=lambda people_waiting_to_be_served: people_waiting_to_be_served.popleft(), 
        enqueue_next_discovery=lambda queue, new_option: queue.append(new_option)
    )

def use_obsessive_but_fast_search_dfs(problem: SearchProblemInterface) -> FinalSearchReport:
    stack_of_papers_on_my_desk = [] # LIFO stack
    return master_graph_search_algorithm(
        mathematical_problem=problem, 
        border_rest_zone=stack_of_papers_on_my_desk, 
        decide_who_advances_now=lambda inbox: inbox.pop(), 
        enqueue_next_discovery=lambda stack, new_element: stack.append(new_element)
    )

def use_artificial_intelligence_type_a_star(maze_problem: SearchProblemInterface) -> FinalSearchReport:
    native_position = maze_problem.initial_state
    
    line_that_favors_important_people = [] # Priority queue
    a_simple_tiebreaker_clock = itertools.count()
    mathematical_attractiveness_towards_goal = maze_problem.remaining_distance_estimation(native_position)
    
    heapq.heappush(line_that_favors_important_people, (mathematical_attractiveness_towards_goal, next(a_simple_tiebreaker_clock), 0.0, native_position))

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

        possible_directions = maze_problem.possible_actions(cell_under_the_magnifying_glass_right_now)
        
        for new_path in possible_directions:
            cell_where_we_will_arrive = maze_problem.resulting_state(cell_under_the_magnifying_glass_right_now, new_path)
            cost_of_this_transition = maze_problem.step_cost(cell_under_the_magnifying_glass_right_now, new_path, cell_where_we_will_arrive)
            
            how_much_money_it_will_cost_to_end_up_here = burned_real_money + cost_of_this_transition
            
            is_a_little_square_never_seen_before = cell_where_we_will_arrive not in purse_of_accumulated_real_cost
            the_new_shortcut_is_cheaper_than_the_old_one = False
            
            if not is_a_little_square_never_seen_before:
                the_new_shortcut_is_cheaper_than_the_old_one = how_much_money_it_will_cost_to_end_up_here < purse_of_accumulated_real_cost[cell_where_we_will_arrive]

            if is_a_little_square_never_seen_before or the_new_shortcut_is_cheaper_than_the_old_one:
                purse_of_accumulated_real_cost[cell_where_we_will_arrive] = how_much_money_it_will_cost_to_end_up_here
                
                estimated_distance_to_goal = maze_problem.remaining_distance_estimation(cell_where_we_will_arrive)
                total_attractiveness_how_good_it_is = how_much_money_it_will_cost_to_end_up_here + estimated_distance_to_goal
                
                heapq.heappush(line_that_favors_important_people, (total_attractiveness_how_good_it_is, next(a_simple_tiebreaker_clock), how_much_money_it_will_cost_to_end_up_here, cell_where_we_will_arrive))
                ariadnes_thread_where_i_come_from[cell_where_we_will_arrive] = cell_under_the_magnifying_glass_right_now

    return FinalSearchReport(False, [], -1.0, general_observation_count)
