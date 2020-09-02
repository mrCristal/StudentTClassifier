from main.processes.baseprocessregression import BaseProcessRegression
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Simple plotter for the BaseRegressionProcess
    """
    def __init__(self, process: BaseProcessRegression):
        self.process = process

    def plot(self, n_samples: int = 0, with_y=False) -> None:
        """
        Plots samples for predictions over current points
        If there are no current training points, the plotted samples will be of the prior from -5 to 5
        If there are training points in 1 dimension then n samples are plotted with the x axis as the training points
        If there are training points in more than 1 dimension then n samples are plotted with the x axis as the
            index of the training points
        :param n_samples: int, the number of points to plot
        :param with_y: bool, specifies if tragets y are to be a plotted as scatter graph on top
        :return: None
        """
        mean = self.process.get_predictions()[0]
        if mean.shape == ():
            xx = np.linspace(-5, 5, 50)
            mean = self.process.get_predictions(xx)[0].flatten()
            mean.resize((xx.shape[0],), refcheck=False)
            std = self.process.get_std(xx).flatten()
            for _ in range(n_samples):
                plt.plot(xx, self.process.get_sample(xx))
            plt.plot(xx, mean, c='red')
            plt.plot(xx, mean + 3 * std, c='black')
            plt.plot(xx, mean - 3 * std, c='black')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.125, color='gray')
            plt.fill_between(xx, mean - 2 * std, mean + 2 * std, alpha=0.25, color='gray')
            plt.fill_between(xx, mean - 1 * std, mean + 1 * std, alpha=0.5, color='gray')

        elif self.process.X.shape[1] > 1:
            xx = np.linspace(0, self.process.X.shape[0]-1, self.process.X.shape[0])
            xx = xx.reshape((-1,))
            mean = mean.flatten()
            std = self.process.get_std()
            std = std.flatten()
            #print(len(xx))
            for _ in range(n_samples):
                plt.plot(self.process.get_sample(self.process.X))
            plt.plot(xx, mean, c='red', label='Predicted Mean')
            plt.plot(xx, mean + 3 * std, c='black')
            plt.plot(xx, mean - 3 * std, c='black')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.125, color='gray')
            plt.fill_between(xx, mean - 2 * std, mean + 2 * std, alpha=0.25, color='gray')
            plt.fill_between(xx, mean - 1 * std, mean + 1 * std, alpha=0.5, color='gray')
            if with_y:
                plt.scatter(xx, self.process.y, c='blue', marker='+', s=150, label='Target value')

        else:
            xx = self.process.X
            xx = xx.reshape((xx.shape[0],))
            mean = mean.flatten()
            std = self.process.get_std()
            std = std.flatten()
            for _ in range(n_samples):
                plt.plot(xx, self.process.get_sample(xx.reshape((-1, 1))))
            plt.plot(xx, mean, c='red', label='Predicted Mean')
            plt.plot(xx, mean + 3 * std, c='black')
            plt.plot(xx, mean - 3 * std, c='black')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.125, color='gray')
            plt.fill_between(xx, mean - 2 * std, mean + 2 * std, alpha=0.25, color='gray')
            plt.fill_between(xx, mean - 1 * std, mean + 1 * std, alpha=0.5, color='gray')
            if with_y:
                plt.scatter(xx, self.process.y, c='blue', marker='+', s=150, label='Target value')
        plt.legend(loc='best')
        plt.show()

    def __call__(self, process: BaseProcessRegression) -> None:
        """
        Updates the process attribute
        :param process: BaseRegressionProcess, the new regression process instance to use
        :return: None
        """
        self.process = process
