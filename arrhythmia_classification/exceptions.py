"""Define exceptions to use within the project to adhere to triceratops.
"""
from typing import Any

class ParameterTypeError(Exception):
    """Define an abstraction of TypeError.
    
    :param expected_type: The expected type for the parameter.
    """

    def __init__(self, expected_type: Any):
        raise TypeError(f"Invalid parameter type. Expected {expected_type}.")
    
class ParameterLengthError(Exception):
    """Define an abstraction of ValueError focussed at asserting a fixed
    size for iterables.
    
    :param elen: The expected length for the parameter.
    """

    def __init__(self, elen: int):
        raise ValueError(f"Incorrect parameter length. Expected {elen}.")
    