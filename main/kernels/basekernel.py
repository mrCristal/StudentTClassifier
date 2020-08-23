import abc
import numpy as np


class BaseKernel(metaclass=abc.ABCMeta):
    """
    Base class for the covariance functions
    """
    def __init__(self, amplitude: float, length_scale: float) -> None:
        self.amplitude = amplitude
        self.length_scale = length_scale

    def set_log_parameters(self, log_amplitude: float, log_length_scale: float) -> None:
        """
        Sets the new log parameters
        Note that only certain values are accepted for each parameter
        :return: None
        """
        self.log_amplitude = log_amplitude
        self.log_length_scale = log_length_scale

    def set_parameters(self, amplitude: float, length_scale: float) -> None:
        """
        Sets the new parameters
        Note that only certain values are accepted for each parameter
        :return: None
        """
        self.amplitude = amplitude
        self.length_scale = length_scale

    @property
    def amplitude(self) -> float:
        return np.exp(self._log_amplitude)

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._log_amplitude = np.log(value)

    @property
    def log_amplitude(self) -> float:
        return self._log_amplitude

    @log_amplitude.setter
    def log_amplitude(self, value: float) -> None:
        self._log_amplitude = np.clip(value, -4, 3).item()

    @property
    def length_scale(self) -> float:
        return np.exp(self._log_length_scale)

    @length_scale.setter
    def length_scale(self, value: float) -> None:
        self._log_length_scale = np.log(value)

    @property
    def log_length_scale(self) -> float:
        return self._log_length_scale

    @log_length_scale.setter
    def log_length_scale(self, value: float) -> None:
        value = np.clip(value, -4, 3).item()
        self._log_length_scale = value

    @property
    def amplitude_squared(self) -> float:
        return self.amplitude ** 2

    def __call__(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        return self.get_cov(X, X_)

    @staticmethod
    def _get_distance(X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        return np.asarray([[np.linalg.norm(x - x_) for x_ in X_] for x in X])

    @abc.abstractmethod
    def get_cov(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        pass

    def get_gradients(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        """
        Returns the gradients of the covariance function w.r.t parameters at for the given data points
        :param X: n x p np.ndarray, data points where the gradient should be evaluated
        :param X_: n x p np.ndarray, data points where the gradient should be evaluated
        :return: n x n np.ndarray, evaluated gradient
        """
        return np.asarray([self.get_dk_damplitude(X, X_), self.get_dK_dlength_scale(X, X_)])

    @abc.abstractmethod
    def get_dK_dlength_scale(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        pass

    def get_dk_damplitude(self, X: np.ndarray, X_: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the covraince function w.r.t. the current amplitude for the given data points
        :param X: n x p np.ndarray, data points where the gradient should be evaluated
        :param X_: n x p np.ndarray, data points where the gradient should be evaluated
        :return: n x np.ndarray, evaluated gradient
        """
        return self.get_cov(X, X_) / self.amplitude * 2

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass

    def __str__(self) -> str:
        return '{t} with amplitude {a} and length scale {l}'.format(t=self.type, a=self.amplitude, l=self.length_scale)
