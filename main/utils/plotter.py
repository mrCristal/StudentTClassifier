from main.processes.baseprocess import BaseProcess
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, process: BaseProcess):
        self.process = process

    def plot(self, n_samples: int = 0, with_y=False) -> None:
        mean = self.process.get_predictions()[0]
        if mean.shape == ():
            xx = np.linspace(0, 50, self.process._X.shape[0])
            xx = xx.reshape((-1,))
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
        if self.process._X.shape[1] >= 2:
            xx = np.linspace(0, self.process._X.shape[0], self.process._X.shape[0])
            xx = xx.reshape((-1,))
            mean = mean.flatten()
            std = self.process.get_std()
            std = std.flatten()
            for _ in range(n_samples):
                plt.plot(self.process.get_sample(self.process._X))
            plt.plot(xx, mean, c='b')
            plt.plot(xx, mean + 3 * std, c='r')
            plt.plot(xx, mean - 3 * std, c='b')
            plt.fill_between(xx, mean - 3 * std, mean + 3 * std, alpha=0.2, color='m')
            if with_y:
                plt.scatter(xx, self.process._y, c='g', marker='+')
            plt.show()

        else:
            xx = self.process._X
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
                plt.scatter(xx, self.process._y, c='g', marker='+')
            plt.show()

    def __call__(self, process: BaseProcess) -> None:
        self.process = process
