from typing import overload

import numpy as np


@overload
def calc_gr(x: np.ndarray, x0: np.ndarray, xctrl: np.ndarray) -> np.ndarray:
    ...


@overload
def calc_gr(x: float, x0: float, xctrl: float) -> float:
    ...


def calc_gr(x: float, x0: float, xctrl: float) -> float:
    """Calculates GR values.

    Examples
    --------
    >>> calc_gr(1, 1, 1)
    nan
    >>> calc_gr(100, 50, 100)
    1.0
    >>> calc_gr(2, 1, 4)
    0.41421356237309515
    """
    enum = np.log2(x / x0)
    denom = np.log2(xctrl / x0)
    return (2 ** (enum / denom)) - 1
