# IA Proyecto 1

Motor de búsqueda que discretiza laberintos a partir de imágenes cuadradas (PNG/BMP) y genera un grafo navegable.

## Requisitos

1. Python 3.11+.
2. Dependencias listadas en `requirements.txt`.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Uso

Ejecute el menú interactivo:

```bash
python -m src.main
```

Opciones disponibles:

1. Resolver el ejemplo `assets/maze.png`.
2. Ingresar la ruta de una imagen personalizada (PNG/BMP).
3. Salir.

El programa carga la imagen, la discretiza (tiles de 20 px con tolerancia 45), construye el grafo y ejecuta búsqueda A*. Finalmente muestra estadísticas básicas y un render ASCII del camino sobre la grilla discreta.

## Notas

- El folder `io/` funciona como paquete Python (gracias a `io/__init__.py`), así que se puede importar con `from io.discretize import ...` sin choques con la librería estándar.
- Ajuste `--tile-size` y `--tolerance` según la resolución del mapa y el ruido de color.

## Task 2.1 - Entrenamiento Red Neuronal (MLP con numpy)

Para entrenar el clasificador RGB con SGD + Backpropagation:

```bash
python -m nn.train --epochs 80 --learning-rate 0.05 --batch-size 32
```

Entradas/salidas:

- Dataset por defecto: `assets/final_data_colors.csv`
- Pesos exportados: `outputs/color_mlp_weights.npz`
- Etiquetas de clase: `outputs/color_mlp_labels.txt`
- Reporte (incluye accuracy y mapeo etiqueta->costo): `outputs/task2_1_report.txt`
