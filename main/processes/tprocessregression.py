from main.processes.baseregressionprocess import BaseRegressionProcess
from main.kernels.basekernel import BaseKernel
import numpy as np
from hyperopt import hp, Trials, fmin, tpe
from collections import OrderedDict
from scipy.optimize import minimize
from scipy.special import digamma as Di, gammaln as lG


class TProcessRegression(BaseRegressionProcess):
    def __init__(self, kernel: BaseKernel, dof: float = 3, noise_scale: float = 0.5, X: np.ndarray = None,
                 y: np.ndarray = None) -> None:
        """
        A class implementing the Student T Process Regression Algorithm
        :param kernel: BaseKernel type, callable, a class implementing the covariance function
        :param noise_scale: float, specifies the noise scale of the process
        :param X: n x p np.ndarray, the training points with dimension p
        :param y: n x 1 np.ndarray, the target points with dimension 1
        :param dof: float, specifies the degrees of freedom for the process
        """
        super().__init__(kernel=kernel, noise_scale=noise_scale, X=X, y=y)
        self.log_dof = np.log(dof)

    @property
    def log_dof(self) -> float:
        return self._log_dof

    @log_dof.setter
    def log_dof(self, value: float) -> None:
        value = np.clip(value, np.log(3), np.log(9e6)).item()
        self._log_dof = value

    @property
    def dof(self) -> float:
        return np.exp(self._log_dof)

    @dof.setter
    def dof(self, value: float) -> None:
        self.log_dof = np.log(value)

    def set_log_params(self, log_amplitude: float, log_length_scale: float, log_noise_scale: float,
                       log_dof: float) -> None:
        """
        Sets the new log hyper parameters for the process
        Updates the covariance function after these are set
        :return: None
        """
        self.kernel.set_log_parameters(log_amplitude=log_amplitude, log_length_scale=log_length_scale)
        self.log_noise_scale = log_noise_scale
        self.log_dof = log_dof
        self.set_covariance()

    def set_params(self, amplitude: float, length_scale: float, noise_scale: float, dof: float) -> None:
        """
        Sets new hyper parameters for the process
        Updates the covariance function after these are set
        :return: None
        """
        self.kernel.set_parameters(amplitude=amplitude, length_scale=length_scale)
        self.noise_scale = noise_scale
        self.dof = dof
        self.set_covariance()

    def get_neg_log_ML(self, log_amplitude: float = None, log_length_scale: float = None,
                       log_noise_scale: float = None, log_dof: float = None) -> float:
        """
        Returns the negative negative log marginal likelihood for a given set of hyper parameters
        If not hyper parameters are passed the current ones will be used
        Note that if hyper parameters are passed in, these will be set for the process and covariance will be updated
        :return: float, the value of the negative log marginal likelihood
        """
        if not log_amplitude:
            log_amplitude = self.kernel.log_amplitude
        if not log_length_scale:
            log_length_scale = self.kernel.log_length_scale
        if not log_noise_scale:
            log_noise_scale = self.log_noise_scale
        if not log_dof:
            log_dof = self.log_dof

        self.set_log_params(log_amplitude, log_length_scale, log_noise_scale, log_dof)
        noise_squared = self.noise_scale ** 2
        v = self.dof
        y = self.y
        X = self.X
        n = X.shape[0]
        K_y = self._cov + noise_squared * np.identity(n)
        K_y_inv = np.linalg.inv(K_y)
        det_K_y = np.linalg.det(K_y)
        p1 = n * np.log((v - 2) * np.pi) + np.log(det_K_y) + (v + n) * np.log(1 + y.T @ K_y_inv @ y / (v - 2))
        p2 = lG(v / 2) - lG((v + n) / 2)
        L = 0.5 * p1 + p2
        return L.item()

    def get_neg_log_ML_gradients(self, log_amplitude: float = None, log_length_scale: float = None,
                                 log_noise_scale: float = None, log_dof: float = None) -> np.ndarray:
        """
        Returns gradients of negative log marginal likelihood w.r.t given hyper parameters
        If arguments are not given, then the current hyper parameters will be used
        The passed in arguments will be set as the new hyper parameters
        :return: 1 x 4 np.ndarray, array of gradient evaluations
                given as [w.r.t amplitude, w.r.t length scale, w.r.t noise scale, w.r.t dof]
        """
        if not log_amplitude:
            log_amplitude = self.kernel.log_amplitude
        if not log_length_scale:
            log_length_scale = self.kernel.log_length_scale
        if not log_noise_scale:
            log_noise_scale = self.log_noise_scale
        if not log_dof:
            log_dof = self.log_dof
        self.set_log_params(log_amplitude, log_length_scale, log_noise_scale, log_dof)
        X = self.X
        dK_damplitude = self.kernel.get_dk_damplitude(X, X)
        dK_dlength_scale = self.kernel.get_dK_dlength_scale(X, X)
        dK_y_dnoise = self.get_dK_y_dnoise()

        dL_damplitude = self.evaluate_neg_log_ML_gradient(dK_damplitude)
        dL_dlength_scale = self.evaluate_neg_log_ML_gradient(dK_dlength_scale)
        dL_dnoise = self.evaluate_neg_log_ML_gradient(dK_y_dnoise)
        dL_dv = self.get_dL_dv()

        return np.asarray([dL_damplitude, dL_dlength_scale, dL_dnoise, dL_dv])

    def get_dL_dv(self) -> float:
        """
        Evaluates the gradient of the negative log marginal likelihood w.r.t degree of freedom hyper parameter
        :return: float, the evaluated gradient
        """
        noise_squared = self.noise_scale ** 2
        v = self.dof
        y = self.y
        X = self.X
        n = X.shape[0]
        K_y_inv = np.linalg.inv(self._cov + noise_squared * np.identity(n))
        v2 = v - 2
        d_m = y.T @ K_y_inv @ y  # mahalnobis distance (slight abuse of names)
        sd_m = 1 + d_m / v2
        dL_dv = 0.5 * (n / v2 + np.log(sd_m) - (v + n) * d_m / (v2 ** 2 * sd_m)) + Di(v / 2) - Di((v + n) / 2)
        return dL_dv.item()

    def evaluate_neg_log_ML_gradient(self, covariance_gradient: np.ndarray) -> float:
        """
        Returns the gradient w.r.t to a hyper parameter given the gradient of the covariance function to the
            hyper parameter
        :param covariance_gradient: 1 x 1 np.ndarray, covariance_gradient: the gradient of the covariance
            function w.r.t the hyper parameter
        :return: float, the evaluated gradient
        """
        y = self.y
        K_y = self._cov + self.noise_scale ** 2 * np.identity(y.shape[0])
        K_y_inv = np.linalg.inv(K_y)
        n = K_y.shape[0]
        v = self.dof
        d_m = y.T @ K_y_inv @ y  # mahalnobis distance (slight abuse of names)
        yK = y.T @ K_y_inv
        v2 = v - 2
        sd_m = 1 + d_m / v2
        grad = 0.5 * (np.trace(K_y_inv @ covariance_gradient) - (v + n) * yK @ covariance_gradient @ yK.T / (v2 * sd_m))
        return grad.item()

    def get_predictions(self, X_: np.ndarray = None) -> tuple:
        """
        Returns the posterior mean, covariance and degree of freedom given the new points
        If no argument is given the prediction for the training points is given
        :param X_: k x p np.ndarray, k points with dimension p to predict
        :return: tuple[k x 1 np.ndarray, k x k np.ndarray, float], the posterior mean, covariance and degree of freedom
        """
        if X_ is None:
            X_ = self.X
        I = np.identity
        X = self.X
        y = self.y
        noise = self.noise_scale ** 2
        v = self.dof
        N2 = y.shape[0]
        k = self.kernel

        K22 = self._cov + noise * I(N2)
        K22_inv = np.linalg.inv(K22)
        K12 = k(X_, X)
        K21 = K12.T
        K11 = k(X_, X_)

        mean = K12 @ K22_inv @ y
        scale = (v + y.T @ K22_inv @ y) / (v + N2) * (K11 - K12 @ K22_inv @ K21)
        v_ = v + N2
        return mean, scale, v_

    def get_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a sample at the given points
        :param X: k x p np.ndarray, points with dimension p where to sample the process
        :return: np.ndarray, a sample of the latent function f for the given input
        """
        mean, scale, v = self.get_predictions(X)
        N = scale.shape[0]
        u = np.random.chisquare(df=v)
        mvn_mean = np.zeros((1, N)).flatten()
        Y = np.random.multivariate_normal(mvn_mean, scale)
        sample = Y / np.sqrt(u / v) + mean.flatten()
        sample = sample.reshape(N, )
        return sample

    def optimise_params(self, number_of_restarts: int, verbose: bool = False, with_BO: bool = True) -> None:
        """
        Optimises the hyper parameters with respect to the negative log marginal likelihood via eitehr BO or BFGS
        Uses the gradients for the BFGS algorithm from scipy
        :param number_of_restarts: int, the number of times to run
        :param verbose: bool, specifies if to output information regarding progression
        :return None
        """

        if with_BO:
            self._BO(number_of_restarts, verbose)
        else:
            self._BFGS(number_of_restarts, verbose)

    def _BO(self, n_iterations: int, verbose: bool) -> None:
        """
        Optimises the hyper parameters via the BO algortithm
        :param n_iterations: int, the number of times to run the BO algorithm
        :param verbose: bool, specifies if to output information regarding the progression of the BO algorithm
        :return:
        """
        space = OrderedDict([('log_amplitude', hp.uniform('log_amplitude', -4, 3)),
                             ('log_length_scale', hp.uniform('log_length_scale', -4, 3)),
                             ('log_noise_scale', hp.uniform('log_noise_scale', -4, 3)),
                             ('log_dof', hp.uniform('log_dof', np.log(3), np.log(9e6)))
                             ])

        def objective(params):
            return self.get_neg_log_ML(**params)

        best_params = fmin(objective, space, trials=Trials(), algo=tpe.suggest, max_evals=n_iterations, verbose=verbose)
        self.set_log_params(**best_params)

    def _BFGS(self, number_of_restarts: int, verbose: bool) -> None:
        """
        Optimises the hyper parameters with respect to the negative log marginal likelihood via the BFGS algorithm
        Runs the algorithm a number of times where starting values for hyper parameters are sampled anew
        :param number_of_restarts: int, the number of times to run the BFGS algorithm
        :param verbose: bool, specifies if to output information regarding the progression of the BFGS algorithm
        :return None
        """

        def get_neg_LML(params: np.ndarray) -> float:
            return self.get_neg_log_ML(params[0].item(), params[1].item(), params[2].item())

        def get_neg_LML_gradients(params: np.ndarray) -> np.ndarray:
            return self.get_neg_log_ML_gradients(params[0].item(), params[1].item(), params[2].item())

        parameter_sets = [[self.kernel.log_amplitude, self.kernel.log_length_scale, self.log_noise_scale, self.log_dof]]

        if number_of_restarts > 1:
            for _ in range(2, number_of_restarts + 1):
                log_amplitude_sample = np.random.uniform(low=-3, high=3)
                log_length_scale_sample = np.random.uniform(low=-3, high=3)
                log_noise_scale_sample = np.random.uniform(low=-3, high=3)
                log_dof_sample = np.random.uniform(low=np.log(3), high=np.log(9e6))
                parameter_sets.append([log_amplitude_sample, log_length_scale_sample, log_noise_scale_sample,
                                       log_dof_sample])
        parameter_sets = np.asarray(parameter_sets)
        best_parameter_set = parameter_sets[0]  # i.e. inital params
        best_NLML_value = get_neg_LML(best_parameter_set)
        count = 1
        for parameter_set in parameter_sets:
            self._print('{}/{} Optimising with {}'.format(count, number_of_restarts, parameter_set), verbose)
            try:
                result = minimize(fun=get_neg_LML, x0=parameter_set, jac=get_neg_LML_gradients, method='BFGS',
                                  options={'disp': False})
            except OverflowError as e:
                self._print('Overflow error, moving to the next set\n', verbose)
                continue
            optimised_parameter_set = result.x
            self._print('Optimised parameters {}'.format(optimised_parameter_set), verbose)
            self._print('NLML value {} \n'.format(result.fun), verbose)

            if result.fun < best_NLML_value:
                best_parameter_set = optimised_parameter_set
                best_NLML_value = result.fun
            count += 1

        self.set_log_params(log_amplitude=best_parameter_set[0].item(), log_length_scale=best_parameter_set[1].item(),
                            log_noise_scale=best_parameter_set[2].item(), log_dof=best_parameter_set[3].item())
        self._print('Final NLML value {} \nand optimised parameters {}'.format(best_NLML_value, best_parameter_set),
                    verbose)

    def get_std(self, X: np.ndarray = None) -> np.ndarray:
        """
        Returns posterior standard deviation based on the new points
        If no argument is given the standard deviation for the training points are given
        Note, the std is not the sqr of the diagonal of the "covariance" of the process. The "covariance" must be scaled
            first by v/(v-2)
        :param X: k x p np.ndarray, points where the standard deviation is to be evaluated
        :return: k x 1 np.ndarray, array of standard deviations
        """
        v = self.dof
        scale = self.get_predictions(X)[1]
        cov = v / (v - 2) * scale
        var = cov.diagonal()
        std = np.sqrt(var)
        return std
