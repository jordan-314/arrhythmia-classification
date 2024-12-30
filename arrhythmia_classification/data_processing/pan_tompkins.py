"""Define the Pan-Tompkins Algorithm for signal processing.
"""
from collections.abc import Iterable

import numpy as np

from .bandpass import Bandpass
from ..exceptions import ParameterTypeError


class PanTompkins:
    """Define the PT algorithm as per the literature.
    
    Algorithm:
        - Apply a bandpass filter to the raw signal.
        - Differentiate the filtered signal.
        - Square the differential signal.
        - Apply an intergration window to the squared signal.

    :param signal: The raw signal to apply the PT algorithm to.
    """

    def __init__(self, signal: Iterable):
        if not isinstance(signal, Iterable):
            raise InvalidParameterError(signal, Iterable)
        
        self._signal = signal

    @property
    def raw_signal(self):
        return self._signal

    def __call__(self,
            lowcut: float = .1,
            highcut: float = 15.,
            filter_order: int = 1
        ):
        """Executes the Pan-Tompkins algorithm."""

        bandpass_filter = Bandpass(self._signal, self._frequency)
        filtered_signal = bandpass_filter(
            lowcut=lowcut,
            highcut=highcut,
            filter_order=filter_order
        )

        diff_signal = self._differentiate(filtered_signal)
        squared_signal = self._square_signal(diff_signal)
        integrated_signal = self._integration_window(squared_signal)

        return integrated_signal

    def _differentiate(self, signal: Iterable):
        """Differentiate the signal, should be bandpass filtered."""

        return np.ediff1d(signal)
    
    def _square_signal(self, signal: Iterable):
        """Square the signal, should be differentiated."""

        return signal ** 2
    
    def _integration_window(self, signal: Iterable):
        """Apply an integration window to the signal."""

        convolve_with = np.ones(int(15 * (self.frequency / 250.)))
        return np.convolve(signal, convolve_with)
