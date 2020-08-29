import numpy as np
from main.kernels.basekernel import BaseKernel
import abc


class BaseRegressionProcess(metaclass=abc.ABCMeta):
    def __init__(self, kernel: BaseKernel, noise_scale: float = 0.5, X: np.ndarray = None,
                 y: np.ndarray = None) -> None:
        self.kernel = kernel
        self.log_noise_scale = np.log(noise_scale)
        if self.data_is_valid(X, y):
            self.initialise_data(X, y)
        elif X is None and y is None:
            self.X = np.asarray([])
            self.y = np.asarray([])
            self._cov = np.asarray([])
        else:
            raise ValueError('Improper data passed in. Make sure both X and y are valid')

    @staticmethod
    def data_is_valid(X: np.ndarray, y: np.ndarray) -> bool:
        if X is not None and y is not None:
            return X.shape[0] == y.shape[0]
        return False

    def initialise_data(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        if not self.data_is_valid(X, y):
            raise ValueError('The passed in data is of invalid format, {X}\n{y}'.format(X=X, y=y))

        self.X = X
        self.y = y
        self.set_covariance()

    def set_covariance(self):
        self._cov = self.kernel(self.X, self.X)

    def set_kernel_parameters(self, amplitude: float, length_scale: float, as_log=True) -> None:
        if as_log:
            self.kernel.set_log_parameters(amplitude, length_scale)
        else:
            self.kernel.set_parameters(amplitude, length_scale)
        self.set_covariance()

    @abc.abstractmethod
    def get_neg_log_ML(self, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def get_neg_log_ML_gradients(self, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def optimise_params(self, number_of_restarts: int, verbose=False) -> None:
        pass

    @abc.abstractmethod
    def get_sample(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_predictions(self, X_: np.ndarray = None) -> tuple:
        pass

    @abc.abstractmethod
    def set_params(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def set_log_params(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def evaluate_neg_log_ML_gradient(self, covariance_gradient: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def get_std(self, X: np.ndarray = None) -> np.ndarray:
        pass

    def get_dK_y_dnoise(self) -> np.ndarray:
        return 2 * self.noise_scale * np.identity(self.X.shape[0])

    @property
    def noise_scale(self) -> float:
        return np.exp(self._log_noise_scale)

    @noise_scale.setter
    def noise_scale(self, value: float) -> None:
        self.log_noise_scale = np.log(value)

    @property
    def log_noise_scale(self) -> float:
        return self._log_noise_scale

    @log_noise_scale.setter
    def log_noise_scale(self, value: float) -> None:
        value = np.clip(value, -4, 3).item()
        self._log_noise_scale = value

    @staticmethod
    def _print(string: str, verbose: bool) -> None:
        if verbose:
            print(string)
