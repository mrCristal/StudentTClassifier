from main.samplers.basemcmc import BaseMCMC
from typing import Callable, Tuple
from scipy.stats import norm
from numpy import ndarray, mean, asarray, min, max, sum, log, copy
from numpy.random import uniform


class HamiltonMCMC(BaseMCMC):
    MAX_ITERATIONS = 20000

    def __init__(self, negative_log_posterior: Callable, jacobian: Callable, starting_position: ndarray, path_len=2,
                 step_size=0.5, n_samples: int = 8000, verbose: bool = False, **nlp_kwargs):
        """
        A Hamiltonian MCMC sampler.
        :parameter negative_log_posterior: callable to evaluate the negative log posterior given a sample
        :parameter jacobian: a callable to evaluate the gradient of the negative log posterior given a sample
        :parameter starting_position: a np.ndarray to specify the starting position of the chain
        :parameter path_len: int, specifying the path length, default is 2
        :parameter step_size: float, specifying the step size, default is 0.1
        :parameter n_samples: int, specifies the number of samples to be taken
        :parameter verbose: bool, specifies if chain progress should be printed
        :parameter nlp_kwargs: specifies any other arguments to pass to the negative_log_posterior callable
        """
        super().__init__(n_samples=n_samples, verbose=verbose)
        self.check_is_callable(negative_log_posterior)
        self.check_is_callable(jacobian)
        self.path_len = path_len
        self.step_size = step_size
        self.starting_position = starting_position
        self.nlp = negative_log_posterior
        self.nlp_kwargs = nlp_kwargs
        self.jacobian = jacobian

    @property
    def path_len(self) -> int:
        return self._path_len

    @path_len.setter
    def path_len(self, value: int) -> None:
        value = max([2, value])
        self._path_len = int(min([value, 50]))

    @property
    def step_size(self) -> float:
        return self._step_size

    @step_size.setter
    def step_size(self, value: float) -> None:
        value = max([0.05, value])
        self._step_size = min([value, self.path_len])

    def _leapfrog(self, q: ndarray, p: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Implements the leapfrog algorithm required for the Hamiltonian MCMC sampler
        :param q: n x 1 np.ndarray, current sample
        :param p: n x 1 np.ndarray, current momentum variables
        :return: tuple[ndarray, ndarray], q and -p after the steps
        """
        q, p = copy(q), copy(p)
        p -= self.step_size * self.jacobian(q) / 2
        for _ in range(int(self.path_len / self.step_size) - 1):
            q += self.step_size * p
            p -= self.step_size * self.jacobian(q)
        q += self.step_size * p
        p -= self.step_size * self.jacobian(q) / 2

        return q, -p

    def estimate(self, extra_samples: int = 0) -> ndarray:
        """
        Returns the samples after running the chain
        :param extra_samples: int, how many more samples to generate on top of self.n_samples
        :return: p x n np.ndarray, matrix of samples where p is the dimension of the parameters and n is the nr of samples
        """
        accepts = []
        momentum = norm(0, 1)
        if not self.samples:
            self.samples.append(self.starting_position)
        self.n_samples += extra_samples
        while len(self.samples) < self.n_samples:
            p = momentum.rvs(size=self.starting_position.shape)
            is_accepted = 0
            q_new, p_new = self._leapfrog(self.samples[-1], p.reshape((-1, 1)))

            current_nlp = self.nlp(self.samples[-1], **self.nlp_kwargs) - sum(momentum.logpdf(p))
            new_nlp = self.nlp(q_new, **self.nlp_kwargs) - sum(momentum.logpdf(p_new))
            if log(uniform(low=0, high=1)) < current_nlp - new_nlp:
                self.samples.append(q_new)
                is_accepted = 1
            else:
                self.samples.append(self.samples[-1])
            self._iteration_count += 1
            accepts.append(is_accepted)
            if len(self.samples) % 200 == 0:
                self._print("{}/{} samples".format(len(self.samples), self.n_samples))
            if self._iteration_count == self.MAX_ITERATIONS:
                print('Chain stopped as it reached maximum number of iterations at {}'.format(self.MAX_ITERATIONS))
                break

        self.acceptance_rate = mean(accepts)
        return asarray(self.samples)
