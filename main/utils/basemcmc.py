import abc
import numpy as np
from typing import Any, Callable


class BaseMCMC(metaclass=abc.ABCMeta):
    MAX_ITERATIONS = 15000
    MIN_ITERATIONS = 2000

    def __init__(self, n_samples: int, verbose: bool):
        """
        Base class for the 2 MCMC samplers used in this project.
        :parameter n_samples: specifies the number of samples to be taken
        :param verbose: boolean to specify whether or not to output progress of sampler
        """
        self.n_samples = n_samples
        self._verbose = verbose
        self.samples = []
        self.acceptance_rate = None
        self._iteration_count = 0

    @abc.abstractmethod
    def estimate(self, n_samples: int = 0) -> np.ndarray:
        """
        Abstract method to be defined in subclasses. Implements the relevant sampling procedure.
        :return: a np.ndarray of size n_samples x n_parameters
        """
        raise NotImplementedError

    def get_sample_mean(self, n: int = 500) -> np.ndarray:
        """
        :param n: which last n samples to average over
        :return: np.ndarray of the mean over the last n samples for all parameters
        """
        return np.asarray(self.samples[-n:]).mean(axis=0)

    def get_sample_var(self, n: int = 500) -> np.ndarray:
        """
        :param n: which last n samples to average over
        :return: np.ndarray of the mean over the last n samples for all parameters
        """
        return np.asarray((self.samples[-n:] - self.get_sample_mean(n))**2).sum(axis=0) * 1/(n-1)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        value = np.max([self.MIN_ITERATIONS, value])
        self._n_samples = value

    def _print(self, text: Any) -> None:
        """
        Custom print function. Used to output chain updates based on the verbose attribute
        :param text: the text to be printed
        """
        if self._verbose:
            print(text, end='\r')

    @staticmethod
    def check_is_callable(function: Callable):
        """
        Method to ensure a given argument is callable. If not a ValueError is raised.
        :param function: argument to be tested
        """
        if not callable(function):
            raise ValueError('{} is not a callable function'.format(function))

    def can_run(self) -> bool:
        return self._iteration_count < self.MAX_ITERATIONS
