from main.kernels.basekernel import BaseKernel
import numpy as np


class OUKernel(BaseKernel):
    def __init__(self, amplitude: float, length_scale: float) -> None:
        super().__init__(amplitude=amplitude, length_scale=length_scale)

    def get_cov(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        a2 = self.amplitude_squared
        l_s = self.length_scale
        return a2 * np.exp(-1 / l_s * self._get_distance(X, X_))

    def get_gradients(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        K = self.get_cov(X, X_)
        d = self._get_distance(X, X_)
        dK_damplitude = K / self.amplitude * 2
        dK_dlength_scale = d / (self.length_scale ** 2) @ K
        return np.asarray([dK_damplitude, dK_dlength_scale])

    @property
    def type(self) -> str:
        return 'Ornstein - Uhlenbeck'
