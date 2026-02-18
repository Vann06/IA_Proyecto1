"""Interfaz de línea de comandos simple para resolver laberintos discretizados."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from core.problem import build_problem_from_grid
from core.search import a_star_search
from maze_io.discretize import CellType, GridRepresentation, discretize_image
from maze_io.image_loader import load_rgb_image

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_IMAGE = ASSETS_DIR / "maze.png"
DEFAULT_TILE_SIZE = 20
DEFAULT_TOLERANCE = 45.0


def main() -> None:
    while True:
        _print_menu()
        choice = input("Seleccione una opción: ").strip()
        if choice == "1":
            _run_solver(DEFAULT_IMAGE)
        elif choice == "2":
            path = input("Ruta de la imagen (PNG/BMP): ").strip()
            if path:
                _run_solver(Path(path))
        elif choice == "3":
            print("Hasta luego!")
            break
        else:
            print("Opción no válida. Intente de nuevo.\n")


def _print_menu() -> None:
    print("\n=== Solucionador de Laberintos ===")
    print("1) Resolver assets/maze.png (demo)")
    print("2) Resolver imagen personalizada")
    print("3) Salir")


def _run_solver(image_path: Path) -> None:
    try:
        print(f"\nCargando imagen: {image_path}")
        image = load_rgb_image(image_path)
        grid_repr = discretize_image(
            image,
            tile_size=DEFAULT_TILE_SIZE,
            tolerance=DEFAULT_TOLERANCE,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error al cargar/discretizar la imagen: {exc}\n")
        return

    problem = build_problem_from_grid(grid_repr)
    result = a_star_search(problem)

    print("\n--- Resultados A* ---")
    if not result.success:
        print("No se encontró un camino al objetivo")
        return

    path_length = len(result.path)
    print(f"Celdas exploradas: {result.explored}")
    print(f"Costo total (movimientos): {result.cost}")
    print(f"Longitud del camino: {path_length}")
    print(f"Camino inicia en: {result.path[0]} y termina en: {result.path[-1]}")
    print(f"Chequeo rápido (cost == pasos): {result.cost == path_length - 1}")

    print("\nVisualización del camino (grid discreto):\n")
    print(_render_path(grid_repr, result.path))


def _render_path(grid: GridRepresentation, path: List[Tuple[int, int]]) -> str:
    mapping = {
        CellType.WALL: "#",
        CellType.FREE: ".",
        CellType.START: "S",
        CellType.GOAL: "G",
    }
    char_rows: List[List[str]] = []
    for row in grid.grid:
        char_rows.append([mapping.get(int(cell), "?") for cell in row])

    goal_set = set(grid.goals)
    for row, col in path:
        if (row, col) == grid.start:
            char_rows[row][col] = "S"
        elif (row, col) in goal_set:
            char_rows[row][col] = "G"
        else:
            char_rows[row][col] = "*"

    return "\n".join("".join(line) for line in char_rows)


if __name__ == "__main__":
    main()
