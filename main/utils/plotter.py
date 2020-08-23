from main.processes.baseregressionprocess import BaseRegressionProcess
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Simple plotter for the BaseRegressionProcess
    """
    def __init__(self, process: BaseRegressionProcess):
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
            plt.plot(xx, mean, c='b')
            plt.plot(xx, mean + 3 * std, c='r')
            plt.plot(xx, mean - 3 * std, c='b')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.2, color='m')
            plt.show()

        elif self.process.X.shape[1] >= 2:
            xx = np.linspace(0, self.process.X.shape[0], self.process.X.shape[0])
            xx = xx.reshape((-1,))
            mean = mean.flatten()
            std = self.process.get_std()
            std = std.flatten()
            for _ in range(n_samples):
                plt.plot(self.process.get_sample(self.process.X))
            plt.plot(xx, mean, c='b')
            plt.plot(xx, mean + 3 * std, c='r')
            plt.plot(xx, mean - 3 * std, c='b')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.2, color='m')
            if with_y:
                plt.scatter(xx, self.process.y, c='g', marker='+')
            plt.show()

        else:
            xx = self.process.X
            xx = xx.reshape((xx.shape[0],))
            mean = mean.flatten()
            std = self.process.get_std()
            std = std.flatten()
            for _ in range(n_samples):
                plt.plot(xx, self.process.get_sample(xx.reshape((-1, 1))))
            plt.plot(xx, mean, c='b')
            plt.plot(xx, mean + 3 * std, c='r')
            plt.plot(xx, mean - 3 * std, c='b')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.2, color='m')
            if with_y:
                plt.scatter(xx, self.process.y, c='g', marker='+')
            plt.show()

    def __call__(self, process: BaseRegressionProcess) -> None:
        """
        Updates the process attribute
        :param process: BaseRegressionProcess, the new regression process instance to use
        :return: None
        """
        self.process = process
