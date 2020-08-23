from main.utils.basemcmc import BaseMCMC
from typing import Callable
from numpy.random import uniform
from numpy import ndarray, mean, asarray, log


class MetropolisMCMC(BaseMCMC):
    def __init__(self, is_random_walk: bool, log_posterior: Callable, proposal: Callable,
                 starting_position: ndarray = None, n_samples: int = 5000, verbose: bool = False, **lp_kwargs):
        """
        A Metropolis MCMC sampler. This means the proposal distribution MUST be symmetric, e.g. Gaussian
        :parameter is_random_walk: bool to specify if proposal distribution q is independent or not
        :parameter lp: a callable to evaluate the log posterior given sample x
        :parameter proposal: a callable, proposal distribution q which may or may not take an argument mean
        :parameter starting_position: a np.ndarray to specify the starting position of the chain
        :parameter n_samples: specifies the number of samples to be taken
        :parameter lp_kwargs: specifies any other arguments to pass to the log_posterior callable
        """
        super().__init__(n_samples=n_samples, verbose=verbose)
        self.check_is_callable(log_posterior)
        self.check_is_callable(proposal)
        self.is_random_walk = is_random_walk
        self.lp = log_posterior
        self.proposal = proposal
        self.starting_position = starting_position
        self.lp_kwargs = lp_kwargs

    def estimate(self, extra_samples: int = 0) -> ndarray:
        """
        Returns the samples after running the chain
        :param extra_samples: int, how many more samples to generate on top of self.n_samples
        :return: p x n np.ndarray, matrix of samples where p is the dimension of the parameters and n is the nr of samples
        """
        accepts = []
        if self.starting_position is not None:
            current_sample = self.starting_position
        else:
            current_sample = self.proposal()  # if no starting_position the proposal must be independent

        self.n_samples += extra_samples
        while len(self.samples) < self.n_samples:
            is_accepted = 0
            if self.is_random_walk:
                new_sample = self.proposal(current_sample)
            else:
                new_sample = self.proposal()
            u = log(uniform(low=0, high=1))
            alpha = self.lp(new_sample, **self.lp_kwargs) - self.lp(current_sample, **self.lp_kwargs)
            r = min(0, alpha)
            if r > u:
                self.samples.append(new_sample)
                current_sample = new_sample
                is_accepted = 1
            else:
                self.samples.append(current_sample)

            self._iteration_count += 1
            accepts.append(is_accepted)
            if len(self.samples) % 200 == 0:
                self._print("{}/{} samples".format(len(self.samples), self._n_samples))
            if self._iteration_count == self.MAX_ITERATIONS:
                print('Chain stopped as it reached maximum number of iterations at {}'.format(self.MAX_ITERATIONS))
                break

        self.acceptance_rate = mean(accepts)
        return asarray(self.samples)
