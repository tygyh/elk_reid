import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_a():
    arr = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
            [[20, 30, 40], [50, 60, 70], [80, 90, 100]],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def image_b():
    arr = np.array(
        [
            [[100, 90, 80], [70, 60, 50], [40, 30, 20]],
            [[95, 85, 85], [75, 65, 55], [45, 35, 25]],
            [[90, 80, 70], [60, 50, 40], [30, 20, 10]],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def x_mask():
    return np.array(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )


@pytest.fixture
def y_mask():
    return np.array(
        [
            [True, False, True],
            [False, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )


@pytest.fixture
def tiny_mask():
    return np.array(
        [
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )


@pytest.fixture
def small_mask():
    return np.array(
        [
            [False, False, False],
            [False, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )
