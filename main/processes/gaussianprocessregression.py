from main.processes.baseprocess import BaseProcess
from main.kernels.basekernel import BaseKernel
import numpy as np
from scipy.optimize import minimize


class GaussianProcessRegression(BaseProcess):
    def __init__(self, kernel: BaseKernel, noise_scale: float = 0.5, X: np.ndarray = None,
                 y: np.ndarray = None) -> None:
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

    def get_sample(self, X: np.ndarray) -> np.ndarray:
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
        grad = 0.5 * np.trace((a @ a.T - K_inv) @ covariance_gradient)
        return grad.item()

    def optimise_params(self, number_of_restarts: int, verbose=False) -> None:
        def get_neg_LML(params: np.ndarray) -> float:
            self.set_log_params(log_amplitude=params[0].item(), log_length_scale=params[1].item(),
                                log_noise_scale=params[2].item())
            return self.get_neg_log_ML()

        def get_neg_LML_gradients(params: np.ndarray) -> np.ndarray:
            self.set_log_params(log_amplitude=params[0].item(), log_length_scale=params[1].item(),
                                log_noise_scale=params[2].item())
            return self.get_neg_log_ML_gradients()

        parameter_sets = [self.kernel.log_amplitude, self.kernel.log_length_scale, self.log_noise_scale]

        if number_of_restarts > 1:
            for _ in range(2, number_of_restarts + 1):
                log_amplitude_sample = np.random.uniform(low=-4, high=4)
                log_length_scale_sample = np.random.uniform(low=-4, high=3)
                log_noise_scale_sample = np.random.uniform(low=-4, high=4)
                parameter_sets.append([log_amplitude_sample, log_length_scale_sample, log_noise_scale_sample])
        parameter_sets = np.asarray(parameter_sets)
        best_parameter_set = parameter_sets[0]  # i.e. inital params
        best_NLML_value = get_neg_LML(best_parameter_set)
        for parameter_set in parameter_sets:
            self._print('Optimising with {}'.format(parameter_set), verbose)
            try:
                result = minimize(fun=get_neg_LML, x0=parameter_set, jac=get_neg_LML_gradients, method='BFGS',
                                  options={'disp': verbose})
            except OverflowError as e:
                self._print('Overflow error, moving to the next set\n', verbose)
                continue
            optimised_parameter_set = result.x
            self._print('Optimised parameters {}'.format(optimised_parameter_set), verbose)
            self._print('NLML value {}'.format(result.fun), verbose)

            if result.fun < best_NLML_value:
                best_parameter_set = optimised_parameter_set
                best_NLML_value = result.fun

        self.set_log_params(log_amplitude=best_parameter_set[0].item(), log_length_scale=best_parameter_set[1].item(),
                            log_noise_scale=best_parameter_set[2].item())
        self._print('Final NLML value {} \nand optimised parameters {}'.format(best_NLML_value, best_parameter_set),
                    verbose)
