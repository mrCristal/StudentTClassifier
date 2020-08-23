from main.kernels.basekernel import BaseKernel
import numpy as np


class OUKernel(BaseKernel):
    """
    Class implementing the Ornstein - Uhlenbeck kernel function
    """
    def __init__(self, amplitude: float, length_scale: float) -> None:
        super().__init__(amplitude=amplitude, length_scale=length_scale)

    def get_cov(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        """
        Returns the covariance as given by the OU algorithm
        :param X: n x p np.ndarray, data points where the covariance should be evaluated
        :param X_: n x p np.ndarray, data points where the covariance should be evaluated
        :return: n x n np.ndarray, covariance matrix
        """
        a2 = self.amplitude_squared
        l_s = self.length_scale
        return a2 * np.exp(-1 / l_s * self._get_distance(X, X_))

    def get_dK_dlength_scale(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the covariance function w.r.t. the length scale for the give data points
        :param X: n x p np.ndarray, data points where the gradient should be evaluated
        :param X_: n x p np.ndarray, data points where the gradient should be evaluated
        :return: n x n np.ndarray, evaluated gradient
        """
        d = self._get_distance(X, X_)
        K = self.get_cov(X, X_)
        dK_dlength_scale = d / (self.length_scale ** 2) * K
        return dK_dlength_scale

    @property
    def type(self) -> str:
        return 'Ornstein - Uhlenbeck'
