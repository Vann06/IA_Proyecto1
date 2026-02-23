"""Intelligent console center. Serves with care and informs in a human way."""

from __future__ import annotations

from pathlib import Path

# Aqui importamos con los nombres descriptivos en ingles!
from core.problem import build_problem_from_maze
from core.search import (
    use_artificial_intelligence_type_a_star, 
    use_relaxed_and_egalitarian_search_bfs, 
    use_obsessive_but_fast_search_dfs
)
from maze_io.discretize import discretize_image
from maze_io.image_loader import load_rgb_image
from viz.draw import save_discretization_overlay, draw_marker_over_original_image, save_path_on_grid

THE_ZONE_OF_MY_FILES = Path(__file__).resolve().parents[1] / "assets"
FOLDER_TO_SAVE_THE_MAGIC = Path(__file__).resolve().parents[1] / "outputs"
DEFAULT_SYSTEM_PHOTO = THE_ZONE_OF_MY_FILES / "maze.png"

LITTLE_CUBE_SIZE = 20
COLOR_CONFIDENCE_DEGREE = 45.0

import numpy as np
from nn.mlp import MLPClassifierNumpy
from nn.dataset import build_terrain_cost_map

def start_software() -> None:
    while True:
        _tell_me_available_options_nicely()
        what_the_user_wants_to_do = input("Type your selection here: ").strip()
        
        if what_the_user_wants_to_do == "1":
            _process_entire_request_step_by_step(DEFAULT_SYSTEM_PHOTO)
            
        elif what_the_user_wants_to_do == "2":
            photo_address = input("Carefully type the path to your image: ").strip()
            if photo_address:
                _process_entire_request_step_by_step(Path(photo_address))
                
        elif what_the_user_wants_to_do == "3":
            photo_address = input("Carefully type the path to your very complex image: ").strip()
            if photo_address:
                _process_entire_request_step_by_step(Path(photo_address), image_is_complicated_so_caution=True)
                
        elif what_the_user_wants_to_do == "4":
            print("See you in the next search!")
            break
        else:
            print("Oops, I'm afraid that's not a menu option. Let's try again.\n")

def _tell_me_available_options_nicely() -> None:
    print("\n========= Robot Maze-Solver =========")
    print("1) I want to use the factory default image.")
    print("2) I have my own image with simple design (20x20 squares).")
    print("3) I have my own image with super demanding design (tiny pixels).")
    print("4) I got bored, I want to close the terminal.")

def _process_entire_request_step_by_step(person_file_url: Path, image_is_complicated_so_caution: bool = False) -> None:
    try:
        print(f"\nNoted! Preparing my lenses to read the file: {person_file_url}")
        photo_with_its_original_colors = load_rgb_image(person_file_url)
        
        if image_is_complicated_so_caution:
            brain_map_of_maze = discretize_image(
                photo_with_its_original_colors,
                tile_size=1,
                tolerance=COLOR_CONFIDENCE_DEGREE,
                is_complex=True,
            )
            actual_tile_size = 1
        else:
            brain_map_of_maze = discretize_image(
                photo_with_its_original_colors,
                tile_size=LITTLE_CUBE_SIZE,
                tolerance=COLOR_CONFIDENCE_DEGREE,
            )
            actual_tile_size = LITTLE_CUBE_SIZE
    except Exception as what_happened:
        print(f"Bad news engineer, something happened with the image file: {what_happened}\n")
        return

    FOLDER_TO_SAVE_THE_MAGIC.mkdir(parents=True, exist_ok=True)
    how_the_photo_is_called_without_extension = person_file_url.stem
    
    # Nueva salida solicitada: la imagen fotográfica base, sin ningún path ni cuadrícula dibujada
    path_for_purely_discrete_photo = FOLDER_TO_SAVE_THE_MAGIC / f"{how_the_photo_is_called_without_extension}_discreto.png"
    draw_marker_over_original_image(
        original_photographic_frame=photo_with_its_original_colors, 
        analyzed_maze_logical_information=brain_map_of_maze, 
        list_of_winning_steps=[], 
        name_of_the_future_exported_file=path_for_purely_discrete_photo, 
        force_window_to_appear=False
    )
    # The original save_discretization_overlay call is removed as per the instruction's implied replacement.
    # The new code block replaces the previous logic for saving the discretized photo.

    print("\nDiscretization ready. Select the Algorithm:")
    print("1) Breadth-First Search (BFS)")
    print("2) Depth-First Search (DFS)")
    print("3) A* Search (Heuristic A-Star)")
    algorithm_choice = input("Algorithm [1-3] > ").strip()
    
    live_cost_predictor = None

    import subprocess
    import sys

    if algorithm_choice == "3":
        print("Loading previously trained MLP Brain to infer dynamic costs...")
        model_path = FOLDER_TO_SAVE_THE_MAGIC / "color_mlp_weights.npz"
        labels_path = FOLDER_TO_SAVE_THE_MAGIC / "color_mlp_labels.txt"
        
        if not (model_path.exists() and labels_path.exists()):
            print("MLP weights not found! Training model on the fly, please wait a few seconds...")
            subprocess.run([sys.executable, "-m", "nn.train", "--epochs", "80"], check=True)
            print("Training complete! Loading weights now...\n")

        if model_path.exists() and labels_path.exists():
            labels = labels_path.read_text(encoding="utf-8").strip().split("\n")
            terrain_cost_map = build_terrain_cost_map(labels)
            
            with np.load(model_path) as data:
                weights = [data[f"w{i}"] for i in range(len(data)//2)]
                biases = [data[f"b{i}"] for i in range(len(data)//2)]
            
            input_size = weights[0].shape[0]
            output_size = weights[-1].shape[1]
            hidden_sizes = tuple(w.shape[1] for w in weights[:-1])
            
            mlp_brain = MLPClassifierNumpy(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
            mlp_brain.weights = weights
            mlp_brain.biases = biases
            
            def create_live_predictor(img: np.ndarray, t_size: int, mlp: MLPClassifierNumpy, lbls: list[str], cost_map: dict[str, int]):
                def predictor(state: tuple[int, int]) -> float:
                    row, col = state
                    y0 = row * t_size
                    y1 = min(y0 + t_size, img.shape[0])
                    x0 = col * t_size
                    x1 = min(x0 + t_size, img.shape[1])
                    
                    tile = img[y0:y1, x0:x1]
                    avg_color = tile.reshape(-1, 3).mean(axis=0)
                    normalized_color = (avg_color / 255.0).astype(np.float64).reshape(1, 3)
                    
                    class_idx = mlp.predict(normalized_color)[0]
                    predicted_label = lbls[class_idx]
                    
                    return float(cost_map.get(predicted_label, 5.0))
                return predictor
                
            live_cost_predictor = create_live_predictor(photo_with_its_original_colors, actual_tile_size, mlp_brain, labels, terrain_cost_map)
        else:
            print("Warning: MLP model or labels could not be generated! Using standard cost (1.0).")

    mathematical_problem = build_problem_from_maze(brain_map_of_maze, cost_fn=live_cost_predictor)
    print("\nSolving maze...")

    if algorithm_choice == "2":
        triumph_report = use_obsessive_but_fast_search_dfs(mathematical_problem)
        name_of_the_algorithm = "DFS"
    elif algorithm_choice == "3":
        triumph_report = use_artificial_intelligence_type_a_star(mathematical_problem)
        name_of_the_algorithm = "A-Star"
    else:
        triumph_report = use_relaxed_and_egalitarian_search_bfs(mathematical_problem)
        name_of_the_algorithm = "BFS"

    print(f"\n--- Results {name_of_the_algorithm} ---")
    if not triumph_report.success_reaching_goal:
        print("A path to the goal was not found")
        return

    total_path_steps = len(triumph_report.history_of_winning_steps)
    print(f"Explored cells before winning: {triumph_report.amount_of_checked_boxes_before_winning}")
    print(f"Wasted energy (moves): {triumph_report.wasted_energy_cost}")
    print(f"Path length: {total_path_steps}")

    path_image_file = FOLDER_TO_SAVE_THE_MAGIC / f"{how_the_photo_is_called_without_extension}_line_{name_of_the_algorithm}.png"
    
    print("\nGenerating the requested graphic display for Task 1.2...")
    draw_marker_over_original_image(
        original_photographic_frame=photo_with_its_original_colors, 
        analyzed_maze_logical_information=brain_map_of_maze, 
        list_of_winning_steps=triumph_report.history_of_winning_steps, 
        name_of_the_future_exported_file=path_image_file, 
        force_window_to_appear=False
    )
    print(f"Graphic image expertly exported to: {path_image_file}\n")

if __name__ == "__main__":
    start_software()
