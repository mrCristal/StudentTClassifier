import numpy as np


class StockRunner:
    def __init__(self, model, X: np.ndarray, y: np.ndarray, prices: np.ndarray, starting_cash: int = 1000,
                 n_training_days: int = 20):
        self.model = model
        self.X = X
        self.y = y
        self.prices = prices
        self.cash = starting_cash
        self.n_training_days = n_training_days
        self.n_prediction_days = 1
        self.portfolio_value = [self.cash]
        self.stock_count = 0
        self.predicted_values = []
        self.decision = []

    def run(self):
        X = self.X
        y = self.y
        prices = self.prices
        n = self.n_training_days
        n_remaining_prediction_days = len(y) - n
        base_index = 0
        model = self.model
        while n_remaining_prediction_days > 0:
            print('Nr days remaining', n_remaining_prediction_days)
            test_X = X[base_index + n + 1].reshape((1, -1))
            train_X = X[base_index:base_index + n]
            train_y = y[base_index:base_index + n]
            model.initialise_data(train_X, train_y)
            model.optimise_params(number_of_restarts=500, verbose=False, with_BO=True)
            pred_y = model.get_predictions(test_X)[0]
            if pred_y > train_y[-1]:
                c = self._buy(prices[base_index + n])
                print('Bought {} stock at {}'.format(c, prices[base_index + n]))
                self.decision.append(1)
            else:
                print('Sold {} stocks at {}'.format(self.stock_count, prices[base_index + n]))
                self._sell(prices[base_index + n])
                self.decision.append(-1)
            self._update_portfolio_value(prices[base_index + n])
            print('Portfolio value at', self.portfolio_value[-1])
            print('=' * 25 + '\n')
            self.predicted_values.append(pred_y)
            n_remaining_prediction_days -= 1
            base_index += 1

    def _buy(self, price: float):
        price *= 1.001
        c = int(self.cash//price)
        self.cash -= c*price
        self.stock_count += c
        return c

        #while self.cash >= price:
        #    self.cash -= price
        #    self.stock_count += 1

    def _sell(self, price: float):
        self.cash += self.stock_count * price * 0.999
        self.stock_count = 0

    def _update_portfolio_value(self, price):
        self.portfolio_value += [self.cash + self.stock_count * price]
