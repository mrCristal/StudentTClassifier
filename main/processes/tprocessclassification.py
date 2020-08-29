import numpy as np
from main.kernels.basekernel import BaseKernel
from hyperopt import hp, Trials, fmin, tpe
from collections import OrderedDict
from numpy.linalg import det, inv
from numpy import log, sum
from main.utils.hamiltonmcmc import HamiltonMCMC as HMC
from scipy.special import loggamma as lG


class TProcessClassification:
    def __init__(self, kernel: BaseKernel, dof: float = 5, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Class implementing the Student Process classification algorithm
        :param kernel: BaseKernel, callable, object implementing the covariance function
        :param dof: float, specifies the degrees of freedom for the class
        :param X: n x p np.ndarray, training points with dimension p
        :param y: n x 1 np.ndarray, target points with dimension 1
        """
        self.log_dof = log(dof)
        self.kernel = kernel
        self.X = np.asarray([])
        self.y = np.asarray([])
        if self.data_is_valid(X, y):
            self.X = X
            self.y = y.reshape((-1, 1))
        self.set_covariance()
        self._f_mode = None
        self._post_cov = None

    @staticmethod
    def data_is_valid(X: np.ndarray, y: np.ndarray) -> bool:
        """
        Checks that the passed in data is valid, specifically that both X and y are not None and the same amount of
            points has been passed in
        :param X: n x p np.ndarray, training points
        :param y: n x 1 np.ndarray, target points
        :return: bool
        """
        if X is not None and y is not None:
            return X.shape[0] == y.shape[0]
        return False

    @property
    def log_dof(self) -> float:
        return self._log_dof

    @log_dof.setter
    def log_dof(self, value: float) -> None:
        value = np.max([value, np.log(3)]).item()
        self._log_dof = value

    @property
    def dof(self) -> float:
        return np.exp(self._log_dof)

    @dof.setter
    def dof(self, value: float) -> None:
        self.log_dof = np.log(value)

    @staticmethod
    def sigmoid(A: np.ndarray) -> np.ndarray:
        """
        Returns the sigmoid of the argument
        :param A: n x 1 np.ndarray, points to push through the sigmoid
        :return: n x 1 np.ndarray
        """
        return 1 / (1 + np.exp(-A))

    def set_log_params(self, log_amplitude: float, log_length_scale: float, log_dof: float) -> None:
        """
        Sets the new log hyper parameters for the process
        Updates the covariance function after these are set
        :return: None
        """
        self.kernel.set_log_parameters(log_amplitude=log_amplitude, log_length_scale=log_length_scale)
        self.log_dof = log_dof
        self.set_covariance()

    def set_covariance(self) -> None:
        """
        Sets the covariance and inverse covariance using the kernel
        """
        self._cov = self.kernel(self.X, self.X) + np.identity(self.X.shape[0]) * np.exp(-1) ** 2
        self._inv_cov = inv(self._cov)

    def get_Hessian(self, f: np.ndarray) -> np.ndarray:
        """
        Returns the hessian of the unnormalised log posterior w.r.t points of the latent function f
        :param f: n x 1 np.ndarray, points of the latent function f where to evaluate the hessian
        :return: n x n np.ndarray, the evaluated hessian
        """
        v = self.dof
        n = self._cov.shape[0]
        K_inv = self._inv_cov
        pi = self.sigmoid(f)
        dlike_df2 = -pi * (1. - pi)
        W = np.diag(-dlike_df2.reshape(-1))
        a = 2. / v * K_inv @ f
        B = 1. + 1. / v * f.T @ K_inv @ f
        dprior_df2 = -(n + v) / 2. * (B ** -1 * 2. / v * K_inv - a.T @ a * B ** -2)
        return -W + dprior_df2

    def get_Jacobian(self, f: np.ndarray) -> np.ndarray:
        """
        Returns the jacobian of the unnormalised log posterior w.r.t points of the latent function f
        :param f: n x 1 np.ndarray, points of the latent function f where to evaluate the jacobian
        :return: n x 1 np.ndarray, the evaluated jacobian
        """
        v = self.dof
        t = (self.y + 1.) / 2.
        n = self._cov.shape[0]
        K_inv = self._inv_cov
        pi = self.sigmoid(f)
        dlike_df = t - pi

        a = 2. / v * K_inv @ f
        B = 1. + 1. / v * f.T @ K_inv @ f
        dprior_df = -(n + v) / 2. * (a / B)

        dpost_df = dlike_df + dprior_df
        return dpost_df

    def _get_neg_Jacobian(self, f: np.ndarray) -> np.ndarray:
        """
        Returns the jacobian of the negative unnormalised log posterior w.r.t points of the latent function f
        :param f: n x 1 np.ndarray, points of the latent function f where to evaluate the jacobian
        :return: n x 1 np.ndarray, the evaluated jacobian
        """
        return -self.get_Jacobian(f)

    def _get_Newton_ratio(self, f: np.ndarray) -> np.ndarray:
        """
        Returns the inv(hessian) @ jacobian of the unnormalised log posterior
        Used in the Newton method to find the mode of latent function f
        :param f: n x 1 np.ndarray, points of the latent function f where to evaluate the ratio
        :return: n x 1 np.ndarray, the evaluated ratio
        """
        v = self.dof
        t = (self.y + 1.) / 2.
        n = self._cov.shape[0]
        K_inv = self._inv_cov
        pi = self.sigmoid(f)
        dlike_df = t - pi
        dlike_df2 = -pi * (1. - pi)
        W = np.diag(-dlike_df2.reshape(-1))

        a = 2. / v * K_inv @ f
        B = 1. + 1. / v * f.T @ K_inv @ f
        dprior_df = -(n + v) / 2. * (a / B)
        dprior_df2 = -(n + v) / 2. * (B ** -1 * 2. / v * K_inv - a.T @ a * B ** -2)

        dpost_df = dlike_df + dprior_df
        dpost_df2 = - W + dprior_df2
        return inv(dpost_df2) @ dpost_df

    def _iterate_f(self, f: np.ndarray) -> np.ndarray:
        """
        Returns new points for the latent function f after performing 1 Newton method iteration
        :param f: n x 1 np.ndarray, starting points in the latent function f
        :return: n x 1 np.ndarray, new points in the latent function f
        """
        return f - self._get_Newton_ratio(f)

    def _get_unnormalised_post(self, f: np.ndarray, as_log=True, negate=False) -> float:
        """
        Evaluates the unnormalised posterior of the process
        :param f: points in the latent function where to evaluate
        :param as_log: specifies if the result should be a log
        :param negate: specifies if to return the negative unnormalised posterior of the process
        :return: float, the evaluated unnormalised posterior of the process
        """
        c = -1 if negate else 1
        log_likelihood = sum(log(self.sigmoid(self.y * f)))
        log_prior = get_multivariate_T_logpdf(f, np.zeros(f.shape), self._cov, self.dof)
        value = (log_prior + log_likelihood).item()
        if as_log:
            return value * c
        return np.exp(value) * c

    def _get_f_at_mode(self) -> np.ndarray:
        """
        Returns the latent function f that maximises the unnormalised log posterior of the process by performing the
            Newton method iterations
        :return: n x 1 np.ndarray, the latent function f maximising the unnormalised log posterior of the process
        """
        f = np.zeros(self.y.shape)
        f_new = self._iterate_f(f)
        count = 0
        while self._get_unnormalised_post(f_new) >= self._get_unnormalised_post(f):
            f = f_new
            f_new = self._iterate_f(f)
            count += 1
            if count == 50:
                break
        self._f_mode = f_new
        return f_new

    def _get_Laplace_LML(self, log_amplitude: float = None, log_length_scale: float = None,
                         log_dof: float = None) -> float:
        """
        Returns the marginal log likelihood as given by the Laplace approximation
        :return: float, log marginal likelihood
        """
        if not log_amplitude:
            log_amplitude = self.kernel.log_amplitude
        if not log_length_scale:
            log_length_scale = self.kernel.log_length_scale
        if not log_dof:
            log_dof = self.log_dof

        self.set_log_params(log_amplitude, log_length_scale, log_dof)
        f = self._get_f_at_mode()
        H = -self.get_Hessian(f)  # + np.identity(self.X.shape[0]) * np.exp(-1)**3
        det_H = det(H)
        if det_H < 0:
            det_H *= -1
        elif det_H == 0:
            det_H = 0.000001
        log_det_H = log(det_H)
        LML = self._get_unnormalised_post(f) - .5 * log_det_H
        return LML

    def optimise_params(self, number_of_evals: int = 500, verbose: bool = True) -> None:
        """
        Optimises the hyper parameters with respect to the negative log marginal likelihood
        Uses the Bayesian Optimisation algorithm from hyperopt
        :param number_of_evals: int, the number of times to run the B.O algorithm
        :param verbose: bool, specifies if to output information regarding the progression of the B.O algorithm
        :return None
        """
        space = OrderedDict([('log_amplitude', hp.uniform('log_amplitude', -3, 3)),
                             ('log_length_scale', hp.uniform('log_length_scale', -3, 3)),
                             ('log_dof', hp.uniform('log_dof', np.log(3), np.log(9e6)))
                             ])

        def objective(params):
            return - self._get_Laplace_LML(**params)

        optimal_params = fmin(objective, space, trials=Trials(), algo=tpe.suggest, max_evals=number_of_evals,
                              verbose=verbose)
        self.set_log_params(**optimal_params)
        self._print('Final NLML value {} \nand optimised parameters {}'.format(-self._get_Laplace_LML(), optimal_params)
                    , verbose)

    def _get_posterior_cov(self) -> np.ndarray:
        """
        Returns the posterior covariance for the normal that approximates the posterior for given latent function f
        :return: n x n np.ndarray, covariance matrix
        """
        if self._f_mode is None:
            f = self._get_f_at_mode()
        else:
            f = self._f_mode
        cov = inv(-self.get_Hessian(f))
        self._post_cov = cov
        return cov

    def sample(self, mean: np.ndarray = None) -> np.ndarray:
        """
        Returns a sample from a Gaussian proposal distribution
        :param mean: n x 1 np.ndarray, specifies the mean parameter of the proposal distribution, if None then
            the mode of f will be used
        :return: n x 1 np.ndarray, sample
        """
        if self._post_cov is None:
            cov = self._get_posterior_cov()
        else:
            cov = self._post_cov
        if mean is None:
            mean = self._get_f_at_mode()
        c = 2.38 ** 2 / (self.y.shape[0] ** 0.85)
        return np.random.multivariate_normal(mean.flatten(), c * cov, check_valid='ignore').reshape(mean.shape)

    def get_predictions(self, X_: np.ndarray = None, verbose=True) -> np.ndarray:
        """
        Returns the probability p( y = 1 | f) given points X_
        :param X_: k x p np.ndarray, points where to return predictions; if None is given the prediction for the
            training points is given
        :param verbose: bool, specifies whether or not to print updates of inference process
        :return: k x 1 np.ndarray, array of probabilities of y = 1 given X_
        """
        n_samplers = 3
        if X_ is None:
            X_ = self.X
        f_max = self._get_f_at_mode()
        sampler_kwargs = {'negative_log_posterior': self._get_unnormalised_post,
                          'jacobian': self._get_neg_Jacobian,
                          'path_len': 5,
                          'step_size': 0.1,
                          'negate': True}
        samplers = [HMC(starting_position=self.sample(f_max), **sampler_kwargs) for _ in range(n_samplers)]
        for i in range(n_samplers):
            self._print(f'Warming up chain nr {i + 1}', verbose)
            samplers[i].estimate()

        while not self.has_converged(samplers, 1000):
            for sampler in samplers:
                sampler.estimate(1000)
            if not all([s.can_run() for s in samplers]):
                break
        f_post = np.mean([s.get_sample_mean(1000) for s in samplers], axis=0)
        k_ = self.kernel(self.X, X_)
        K_inv = self._inv_cov
        return self.sigmoid(k_.T @ K_inv @ f_post)

    @staticmethod
    def has_converged(samplers: list, n: int) -> bool:
        """
        Checks if the MCMC chains have converged as per the Rubin and Gelman metric R_hat, for each parameter
        :param samplers: list, list of samplers implementing the BaseMCMC interface
        :param n: int, the last n samples over which to take the mean
        :return: bool, whether or not the chains have converged as per the
        """
        means = [s.get_sample_mean(n) for s in samplers]
        var = [s.get_sample_var(n) for s in samplers]
        big_mean = np.mean(means, axis=0)
        W = np.mean(var, axis=0)
        B = np.sum([(m - big_mean) ** 2 for m in means], axis=0) * n / (len(samplers) - 1)
        return (np.sqrt((W + 1 / n * (B - W)) / W) < 1.1).all()

    @staticmethod
    def _print(string: str, verbose: bool) -> None:
        """
        Custom print function, printing text depending on the verbose parameter
        :param string: str, string to be printed
        :param verbose: bool, specifies if string is to be printed
        :return: None
        """
        if verbose:
            print(string)


def get_multivariate_T_logpdf(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, dof: float) -> np.ndarray:
    """
    Evaluates the log pdf of the multivariate student T distribution based on the arguments
    :param x: n x 1 np.ndarray, x points
    :param mean: n np.ndarray, repsecitve means of the x points
    :param scale: n x n np.ndarray, scale matrix of the points x
    :param dof: float, degree of freedom
    :return: n x 1 np.ndarray, array with the evaluated log pdf at points x
    """
    n = mean.shape[0]
    p1 = n * log(dof * np.pi) + log(det(scale)) + (n + dof) * log(1 + 1. / dof * (x - mean).T @ inv(scale) @ (x - mean))
    p2 = lG((dof + n) / 2.) - lG(dof / 2.)
    return -0.5 * p1 + p2
