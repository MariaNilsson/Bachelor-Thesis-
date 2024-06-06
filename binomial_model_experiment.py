import numpy as np
from Final.binomial_model import *


def binomial_delta_experiment_error(path, opt_price, sigma, r, strike, T, boundary_prices):
    """
    Function to calculate the hedging error using the binomial-model.
    :param path: path of stock prices
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: strike price
    :param T: time to maturity
    :param boundary_prices: prices of the early exercise boundary determined by the binomial model
    :return: the hedging error
    """
    # number of total steps
    num_steps = len(path) - 1
    # length of each time step
    dt = T / num_steps

    # defining vector of zeroes to store delta for each step
    deltas = np.zeros(num_steps)
    # defining vector of zeroes to store the risk-free investment
    B = np.zeros(num_steps)

    # for time t=0
    S_0, V_0 = binomial_values_dt(path[0], sigma, r, strike, dt, num_steps)
    deltas[0] = binomial_delta(S_0, V_0)
    B[0] = opt_price - deltas[0] * path[0]

    for i in range(1, num_steps):
        # if the option is exercised
        if path[i] <= boundary_prices[i-1]:
            payoff = np.maximum(strike - path[i], 0)
            port_value = np.exp(r * dt) * B[i - 1] + deltas[i - 1] * path[i]
            error = payoff - port_value
            error = error * np.exp(r * (num_steps - i) * dt)
            break

        # if the option has not been exercised
        else:
            if num_steps - i < 10:
                num_steps_1 = 10
                dt_1 = (T - i * dt) / 10
            else:
                num_steps_1 = num_steps - i
                dt_1 = dt
            S, V = binomial_values_dt(path[i], sigma, r, strike, dt_1, num_steps_1)
            deltas[i] = binomial_delta(S, V)
            B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]
            if i == num_steps - 1:
                payoff = np.maximum(strike - path[i + 1], 0)
                port_value = np.exp(r * dt) * B[i] + deltas[i] * path[i + 1]
                error = payoff - port_value

    return error
