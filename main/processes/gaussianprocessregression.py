from main.processes.baseprocess import BaseProcess
from main.kernels.basekernel import BaseKernel
import numpy as np
from scipy.optimize import minimize


class GaussianProcessRegression(BaseProcess):
    def __init__(self, kernel: BaseKernel, noise_scale: float = 0.5, X: np.ndarray = None,
                 y: np.ndarray = None) -> None:
        super().__init__(kernel=kernel, noise_scale=noise_scale, X=X, y=y)

    def get_neg_log_ML(self, log_amplitude: float = None, log_length_scale: float = None,
                       log_noise_scale: float = None) -> float:
        if not log_amplitude:
            log_amplitude = self.kernel.log_amplitude
        if not log_length_scale:
            log_length_scale = self.kernel.log_length_scale
        if not log_noise_scale:
            log_noise_scale = self.log_noise_scale

        self.set_log_params(log_amplitude, log_length_scale, log_noise_scale)
        noise_squared = self.noise_scale ** 2
        y = self._y
        X = self._X
        n = X.shape[0]
        K_y = self._cov + noise_squared * np.identity(n)
        K_y_inv = np.linalg.inv(K_y)
        log_det_K_y = np.log(np.linalg.det(K_y))
        L = 0.5 * (y.T @ K_y_inv @ y + log_det_K_y + n * np.log(2 * np.pi))
        return L.item()

    def get_sample(self, X: np.ndarray) -> np.ndarray:
        mean, cov = self.get_predictions(X)
        mean = np.asarray(mean)
        mean.resize((cov.shape[0],), refcheck=False)
        return np.random.multivariate_normal(mean.flatten(), cov)

    def get_predictions(self, X_: np.ndarray = None) -> tuple:
        if X_ is None:
            X_ = self._X
        k = self.kernel
        X = self._X
        y = self._y
        noise = self.noise_scale ** 2
        K11 = self._cov + noise * np.identity(X.shape[0])
        K11_inv = np.linalg.inv(K11)
        K12 = k(X, X_)
        K21 = K12.T
        K22 = k(X_, X_)

        mean = K21 @ K11_inv @ y
        cov = K22 - K21 @ K11_inv @ K12
        return mean, cov

    def get_neg_log_ML_gradients(self, log_amplitude: float = None, log_length_scale: float = None,
                                 log_noise_scale: float = None) -> np.ndarray:
        if not log_amplitude:
            log_amplitude = self.kernel.log_amplitude
        if not log_length_scale:
            log_length_scale = self.kernel.log_length_scale
        if not log_noise_scale:
            log_noise_scale = self.log_noise_scale
        self.set_log_params(log_amplitude, log_length_scale, log_noise_scale)

        X = self._X
        dK_damplitude = self.kernel.get_dk_damplitude(X, X)
        dK_dlength_scale = self.kernel.get_dK_dlength_scale(X, X)
        dK_y_dnoise = self.get_dK_y_dnoise()

        dL_damplitude = self.evaluate_neg_log_ML_gradient(dK_damplitude)
        dL_dlength_scale = self.evaluate_neg_log_ML_gradient(dK_dlength_scale)
        dL_dnoise = self.evaluate_neg_log_ML_gradient(dK_y_dnoise)

        return np.asarray([dL_damplitude, dL_dlength_scale, dL_dnoise])

    def evaluate_neg_log_ML_gradient(self, covariance_gradient: np.ndarray) -> float:
        y = self._y
        K_y = self._cov + self.noise_scale ** 2 * np.identity(y.shape[0])
        K_y_inv = np.linalg.inv(K_y)
        a = K_y_inv @ y
        grad = - 0.5 * np.trace((a @ a.T - K_y_inv) @ covariance_gradient)
        return grad.item()

    def optimise_params(self, number_of_restarts: int, verbose=False) -> None:
        def get_neg_LML(params: np.ndarray) -> float:
            return self.get_neg_log_ML(params[0].item(), params[1].item(), params[2].item())

        def get_neg_LML_gradients(params: np.ndarray) -> np.ndarray:
            return self.get_neg_log_ML_gradients(params[0].item(), params[1].item(), params[2].item())

        parameter_sets = [[self.kernel.log_amplitude, self.kernel.log_length_scale, self.log_noise_scale]]

        if number_of_restarts > 1:
            for _ in range(2, number_of_restarts + 1):
                log_amplitude_sample = np.random.uniform(low=-3, high=3)
                log_length_scale_sample = np.random.uniform(low=-3, high=3)
                log_noise_scale_sample = np.random.uniform(low=-3, high=3)
                parameter_sets.append([log_amplitude_sample, log_length_scale_sample, log_noise_scale_sample])
        parameter_sets = np.asarray(parameter_sets)
        best_parameter_set = parameter_sets[0]  # i.e. inital params
        best_NLML_value = get_neg_LML(best_parameter_set)
        for parameter_set in parameter_sets:
            self._print('Optimising with {}'.format(parameter_set), verbose)
            try:
                result = minimize(fun=get_neg_LML, x0=parameter_set, jac=get_neg_LML_gradients, method='BFGS',
                                  options={'disp': False})  # verbose})
            except OverflowError as e:
                self._print('Overflow error, moving to the next set\n', verbose)
                continue
            optimised_parameter_set = result.x
            self._print('Optimised parameters {}'.format(optimised_parameter_set), verbose)
            self._print('NLML value {} \n'.format(result.fun), verbose)

            if result.fun < best_NLML_value:
                best_parameter_set = optimised_parameter_set
                best_NLML_value = result.fun

        self.set_log_params(log_amplitude=best_parameter_set[0].item(), log_length_scale=best_parameter_set[1].item(),
                            log_noise_scale=best_parameter_set[2].item())
        self._print('Final NLML value {} \nand optimised parameters {}'.format(best_NLML_value, best_parameter_set),
                    verbose)

    def set_params(self, amplitude: float, length_scale: float, noise_scale: float) -> None:
        self.kernel.set_parameters(amplitude=amplitude, length_scale=length_scale)
        self.noise_scale = noise_scale
        self.set_covariance()

    def set_log_params(self, log_amplitude: float, log_length_scale: float, log_noise_scale: float) -> None:
        self.kernel.set_log_parameters(log_amplitude=log_amplitude, log_length_scale=log_length_scale)
        self.log_noise_scale = log_noise_scale
        self.set_covariance()

    def get_std(self, X: np.ndarray = None) -> np.ndarray:
        cov = self.get_predictions(X)[1]
        var = cov.diagonal()
        std = np.sqrt(var)
        return std
