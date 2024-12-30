"""Define the continuous wavelet transform methods used in processing the
signals into 2D images.
"""
from collections.abc import Iterable

import numpy as np
import pywt
from skimage.transform import resize

from ..exceptions import ParameterTypeError, ParameterLengthError


def _type_checking(signal, maximum_width, image_size) -> None:
    """Method to perform common type checking for wavelet transforms."""

    if not isinstance(signal, Iterable):
        raise ParameterTypeError(signal, Iterable)
    
    if not isinstance(maximum_width, int):
        raise ParameterTypeError(maximum_width, int)
    
    if not isinstance(image_size, Iterable):
        raise ParameterTypeError(signal, Iterable)

    expected_length = 2
    if not len(image_size) == expected_length:
        raise ParameterLengthError(expected_length)


def complex_morlet_transform(
    signal: Iterable,
    maximum_width: int = 128,
    image_size: Iterable[int, int] = (128, 128)
):
    """Define the complex morlet transform.
    
    :param signal: The signal to transfrom.
    """

    _type_checking(signal, maximum_width, image_size)

    wavelet = "cmor1.5-1"

    widths = np.arange(1, maximum_width + 1)
    img, _ = pywt.cwt(signal, widths, wavelet)
    img = resize(abs(img), image_size)

    return img


def mexican_hat_cwt(
    signal: Iterable,
    maximum_width: int = 32,
    image_size: tuple[int, int] = (128, 128)
):
    """Define the mexican hat (or Ricker wavelet) transform.
    
    :param signal: The signal to transfrom.
    """

    _type_checking(signal, maximum_width, image_size)

    wavelet = "mexh"

    widths = np.arange(1, maximum_width + 1)
    img, _ = pywt.cwt(signal, widths, wavelet)
    img = resize(img, image_size)

    return img
