"""Small demo CLI to load a maze image and print its discrete grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from io.discretize import CellType, discretize_image
from io.image_loader import load_rgb_image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Ruta al archivo PNG/BMP del laberinto",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=20,
        help="Tamaño (en px) de cada tile para discretizar",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=45.0,
        help="Tolerancia en la distancia de color para clasificar celdas",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Imprime una versión ASCII del grid resultante",
    )
    return parser.parse_args()


def _ascii_render(grid, mapping: Dict[int, str]) -> str:
    lines = []
    for row in grid:
        cells = [mapping.get(int(cell), "?") for cell in row]
        lines.append("".join(cells))
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    image = load_rgb_image(args.image)
    grid_repr = discretize_image(image, tile_size=args.tile_size, tolerance=args.tolerance)

    mapping = {
        CellType.WALL: "#",
        CellType.FREE: ".",
        CellType.START: "S",
        CellType.GOAL: "G",
    }

    print(f"Imagen: {args.image}")
    print(f"Grid: {grid_repr.grid.shape[0]} filas x {grid_repr.grid.shape[1]} columnas")
    print(f"Inicio: {grid_repr.start}")
    print(f"Goals: {grid_repr.goals}")

    if args.ascii:
        print("\nASCII grid:\n")
        print(_ascii_render(grid_repr.grid, mapping))


if __name__ == "__main__":  # pragma: no cover
    main()
