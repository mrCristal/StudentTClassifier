import numpy as np
from main.kernels.basekernel import BaseKernel
import abc


class BaseProcess(metaclass=abc.ABCMeta):
    def __init__(self, kernel: BaseKernel, noise_scale: float = 0.5, X: np.ndarray = None, y: np.ndarray = None) -> None:
        self.kernel = kernel
        self.noise_scale = noise_scale
        if X and y:
            self.initialise_data(X, y)
        elif not (X and y):
            self._X = np.asarray([])
            self._y = np.asarray([])
            self._cov = np.asarray([])
        else:
            raise ValueError('Improper data passed in. Make sure both X and y are valid')

    @staticmethod
    def _data_is_valid(X: np.ndarray, y: np.ndarray):
        if 1 not in y.shape:
            return False
            #raise ValueError('y must be one dimensional')
        if X.shape[0] != y.shape[0]:
            return False
        return True

    def initialise_data(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        if not self._data_is_valid(X, y):
            raise ValueError('The passed in data is of invalid format, {X}\n{y}'.format(X=X, y=y))

        self._X = X
        self._y = y
        self.set_covariance()

    def set_covariance(self):
        self._cov = self.kernel(self._X, self._X)

    def set_kernel_parameters(self, amplitude: float, length_scale: float, as_log=True) -> None:
        if as_log:
            self.kernel.set_log_parameters(amplitude, length_scale)
        else:
            self.kernel.set_parameters(amplitude, length_scale)
        self.set_covariance()

    @abc.abstractmethod
    def get_neg_log_ML(self):
        pass

    @abc.abstractmethod
    def get_neg_log_ML_gradients(self):
        pass

    @abc.abstractmethod
    def optimise_params(self, verbose=False):
        pass

    @abc.abstractmethod
    def sample(self, X:np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_predictions(self, X_) -> tuple:
        pass
