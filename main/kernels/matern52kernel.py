from main.kernels.basekernel import BaseKernel
import numpy as np
from numpy import sqrt as sr


class Matern52Kernel(BaseKernel):
    """
    Classs implementing the Matern 5/2 kernel function
    """
    def __init__(self, amplitude: float, length_scale: float) -> None:
        super().__init__(amplitude=amplitude, length_scale=length_scale)

    def get_cov(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        """
        Returns the covariance as given by the Matern algorithm
        :param X: n x p np.ndarray, data points where the covariance should be evaluated
        :param X_: n x p np.ndarray, data points where the covariance should be evaluated
        :return: n x n np.ndarray, covariance matrix
        """
        a2 = self.amplitude_squared
        l_s = self.length_scale
        d = self._get_distance(X, X_)
        m = sr(5) * d / l_s
        m2 = 5 / 3 * d ** 2 / (l_s ** 2)
        return a2 * (1 + m + m2) * np.exp(-m)

    def get_dK_dlength_scale(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the covariance function w.r.t. the length scale for the give data points
        :param X: n x p np.ndarray, data points where the gradient should be evaluated
        :param X_: n x p np.ndarray, data points where the gradient should be evaluated
        :return: n x n np.ndarray, evaluated gradient
        """
        d = self._get_distance(X, X_)
        l_s = self.length_scale
        m = sr(5) * d / l_s
        dK_dlength_scale = d ** 2 * self.amplitude_squared * (5 * l_s + 5 ** 1.5 * d) * np.exp(-m) / (3 * l_s ** 4)
        return dK_dlength_scale

    @property
    def type(self) -> str:
        return 'Matern 5/2'
