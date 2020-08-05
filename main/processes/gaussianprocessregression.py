from main.processes.baseprocess import BaseProcess
from main.kernels.basekernel import BaseKernel
import numpy as np
from scipy.optimize import minimize


class GaussianProcessRegression(BaseProcess):
    def __init__(self, kernel: BaseKernel, noise_scale: float = 0.5, X: np.ndarray = None, y: np.ndarray = None) -> None:
        super().__init__(kernel=kernel, noise_scale=noise_scale, X=X, y=y)

    def get_neg_log_ML(self) -> float:
        noise_squared = self.noise_scale ** 2
        y = self._y
        X = self._X
        n = X.shape[0]
        K_y = self.kernel(X, X) + noise_squared * np.identity(n)
        K_y_inv = np.linalg.inv(K_y)
        log_det_K_y = np.log(np.linalg.det(K_y))
        neg_ML = 0.5 * (y.T @ K_y_inv @ y + log_det_K_y + n * np.log(2 * np.pi))
        return neg_ML.item()

    def get_sample(self, X:np.ndarray) -> np.ndarray:
        mean, cov = self.get_predictions(X)
        mean = mean.flatten()
        return np.random.multivariate_normal(mean, cov)

    def get_predictions(self, X_) -> tuple:
        k = self.kernel
        X = self._X
        y = self._y
        noise = self.noise_scale ** 2
        K11 = self._cov + noise * np.identity(X.shape[0])
        K11_inv = np.linalg.inv(K11)
        K21 = k(X_, X)
        K12 = K21.T
        K22 = k(X_, X_)

        mean = (K21 @ K11_inv @ y).reshape(-1, 1)
        cov = K22 - K21 @ K11_inv @ K12
        return mean, cov

    def get_neg_log_ML_gradients(self) -> np.ndarray:
        X = self._X
        dK_damplitude = self.kernel.get_dk_damplitude(X, X)
        dK_dlength_scale = self.kernel.get_dK_dlength_scale(X, X)
        dK_y_dnoise = self.get_dK_y_dnoise()

        dL_damplitude = self.evaluate_neg_log_ML_gradient(dK_damplitude)
        dL_dlength_scale = self.evaluate_neg_log_ML_gradient(dK_dlength_scale)
        dL_dnoise = self.evaluate_neg_log_ML_gradient(dK_y_dnoise)

        return np.asarray([dL_damplitude, dL_dlength_scale, dL_dnoise])

    def evaluate_neg_log_ML_gradient(self, covariance_gradient: np.ndarray) -> float:
        K = self._cov
        y = self._y
        K_inv = np.linalg.inv(K)
        a = K_inv @ y
        grad = 0.5 * np.trace((a@a.T - K_inv) @ covariance_gradient)
        return grad.item()

    def set_params(self, amplitude, length_scale, noise_scale):
        self.kernel.set_parameters(amplitude=amplitude, length_scale=length_scale)
        self.noise_scale = noise_scale

    def set_log_params(self, log_amplitude, log_length_scale, log_noise_scale):
        self.kernel.set_log_parameters(log_amplitude=log_amplitude, log_length_scale=log_length_scale)
        self.log_noise_scale = log_noise_scale




