from main.kernels.basekernel import BaseKernel
import numpy as np


class GaussianKernel(BaseKernel):
    def __init__(self, amplitude: float, length_scale: float) -> None:
        super().__init__(amplitude=amplitude, length_scale=length_scale)

    def get_cov(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        a2 = self.amplitude_squared
        l_s2 = self.length_scale**2
        d2 = self._get_distance(X, X_) ** 2
        return a2 * np.exp(-0.5 * d2 / l_s2)

    def get_gradients(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        return np.asarray([self.get_dk_damplitude(X, X_), self.get_dK_dlength_scale(X, X_)])

    def get_dK_dlength_scale(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        d2 = self._get_distance(X, X_) ** 2
        K = self.get_cov(X, X_)
        dK_dlength_scale = self.amplitude_squared * self.length_scale ** (-3) * d2 @ K
        return dK_dlength_scale

    @property
    def type(self) -> str:
        return 'Gaussian'
