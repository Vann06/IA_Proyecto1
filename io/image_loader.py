# Cargar la imagen a un np array 

from pathlib import Path
from typing import Union

import matplotlib.image as mpimg
import numpy as np


_SUPPORTED_EXTENSIONS = {".png", ".bmp"}


def load_rgb_image(image_path: Union[str, Path]) -> np.ndarray:
    """Cargar PNG/BMP y retornar un RGB numpy array en formato uint8"""
    path = Path(image_path).expanduser().resolve()
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. Expected one of {_SUPPORTED_EXTENSIONS}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = mpimg.imread(path)
    if image.ndim == 2:  # grayscale
        image = np.stack((image,) * 3, axis=-1)
    if image.shape[2] == 4:  
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    return image
