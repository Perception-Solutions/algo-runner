from typing import Any
from nptyping import NDArray

import numpy as np

__all__ = ["rgb2labels"]


def rgb2labels(
    image: NDArray[(Any, Any, 3), np.int32]
) -> NDArray[(Any, Any), np.int32]:
    """
    Converts an RGB image to a 2D array of labels

    :param image: an RGB image
    :return: a NumPy array where different numbers correspond to different colors of the image
    """
    reshaped = image.reshape(-1, image.shape[2])
    labeled = reshaped[:, 0] * 256**2 + reshaped[:, 1] * 256 + reshaped[:, 2]
    return labeled
