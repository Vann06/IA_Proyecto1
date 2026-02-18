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
