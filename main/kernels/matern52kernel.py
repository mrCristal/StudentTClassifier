from main.kernels.basekernel import BaseKernel
import numpy as np
from numpy import sqrt as sr


class Matern52Kernel(BaseKernel):
    def __init__(self, amplitude: float, length_scale: float) -> None:
        super().__init__(amplitude=amplitude, length_scale=length_scale)

    def get_cov(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        a2 = self.amplitude_squared
        l_s = self.length_scale
        d = self._get_distance(X, X_)
        m = sr(5) * d / l_s
        return a2 * (1 + m + m ** 2 * 1 / 3) @ np.exp(-m)

    def get_gradients(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        K = self.get_cov(X, X_)
        d = self._get_distance(X, X_)
        dK_damplitude = K / self.amplitude * 2
        l_s = self.length_scale
        m = sr(5) * d / l_s
        dK_dlength_scale = d ** 2 * self.amplitude_squared * (5 * l_s + 5 ** 1.5 * d) * np.exp(-m) / (3 * l_s ** 4)

        return np.asarray([dK_damplitude, dK_dlength_scale])

    @property
    def type(self) -> str:
        return 'Matern 5/2'
