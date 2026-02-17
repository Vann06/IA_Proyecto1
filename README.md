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

## Demo rápida

`assets/maze.png` sirve como caso mínimo para validar el pipeline.

```bash
python -m src.main --image assets/maze.png --tile-size 20 --tolerance 55 --ascii
```

El comando imprime:

- Dimensiones del grid discreto.
- Coordenadas de inicio (rojo) y metas (verde).
- Render ASCII opcional para inspección rápida.

## Notas

- El folder `io/` funciona como paquete Python (gracias a `io/__init__.py`), así que se puede importar con `from io.discretize import ...` sin choques con la librería estándar.
- Ajuste `--tile-size` y `--tolerance` según la resolución del mapa y el ruido de color.
