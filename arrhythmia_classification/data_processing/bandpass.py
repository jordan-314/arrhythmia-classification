"""Define a bandpass filter class.
"""
from collections.abc import Iterable

from scipy import signal

from ..exceptions import InvalidParameterError

class Bandpass:
    """Define a bandpass filter leveraging scipy.
    
    :param signal: An iterable, e.g. list or numpy.array containing the signal
        from a single source.
    :param frequency: The frequency of the signal data in Hz.
    """

    def __init__(self, signal: Iterable, frequency: float):
        if not isinstance(signal, Iterable):
            raise InvalidParameterError(signal, Iterable)
        
        if not isinstance(frequency, (float, int)):
            raise InvalidParameterError(frequency, float)
        
        self._signal = signal
        self._frequency = frequency

    @property
    def raw_signal(self):
        return self._signal

    def __call__(self, lowcut: float, highcut: float, filter_order: int):
        """Calls the bandpass filter.
        
        :param lowcut: The lowcut frequency in Hz.
        :param highcut: The highcut frequency in Hz.
        :param filter_order: The bandpass filter_order to pass to scipy's
            filters.
        """

        if not isinstance(lowcut, (float, int)):
            raise InvalidParameterError(lowcut, float)
        
        if not isinstance(highcut, (float, int)):
            raise InvalidParameterError(highcut, float)
        
        if not isinstance(filter_order, int):
            raise InvalidParameterError(filter_order, int)
        
        filtered_signal = self._bandpass(lowcut, highcut, filter_order)
        return filtered_signal

    def _bandpass(self, lowcut, highcut, filter_order):
        """The defintion for the bandpass process."""

        nyquist_freq = 0.5 * self._frequency

        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq

        numerator, denominator = signal.butter(filter_order, [low, high], btype="band")
        filtered_signal = signal.lfilter(numerator, denominator, self._signal)
        return filtered_signal
