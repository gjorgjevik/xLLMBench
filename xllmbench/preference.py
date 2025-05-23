from enum import Enum

import numpy as np


class PreferenceFunction:
    @staticmethod
    def linear(x: np.ndarray, *args: list) -> np.ndarray:
        q = args[0][0]  # min
        p = args[0][1]  # max

        x_linear = x.copy()

        x_linear[x <= q] = 0.0
        x_linear[x > p] = 1.0
        x_linear[(q < x) & (x <= p)] = (x[(q < x) & (x <= p)] - q) / (p - q)

        return x_linear

    @staticmethod
    def usual(x: np.ndarray, *args: list) -> np.ndarray:
        x_usual = np.array(np.greater(x, 0), dtype=int)
        return x_usual

    @staticmethod
    def gaussian(x: np.ndarray, *args: list) -> np.ndarray:
        s = args[0][0]

        d = 2 * pow(s, 2)
        x_square = np.square(x)
        x_gaussian = 1.0 - np.exp(-1.0 * x_square / d)
        x_gaussian[x <= 0.0] = 0.0

        return x_gaussian


class PreferenceFunctionEnum(Enum):
    USUAL: str = 'usual',
    LINEAR: str = 'linear',
    GAUSSIAN: str = 'gaussian',

    @staticmethod
    def get_function(value: str):
        if value == PreferenceFunctionEnum.USUAL:
            return PreferenceFunction.usual
        elif value == PreferenceFunctionEnum.LINEAR:
            return PreferenceFunction.linear
        elif value == PreferenceFunctionEnum.GAUSSIAN:
            return PreferenceFunction.gaussian
        else:
            return None
