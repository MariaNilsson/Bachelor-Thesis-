import numpy as np
from Final.black_scholes import *


def BS_delta_experiment_error(path, opt_price, sigma, r, strike, T, boundary_prices):
    """
    Function to calculate the hedging error using the Black-Scholes formula.
    :param path: path of stock prices
    :param opt_price: time 0 price of the option
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
    deltas[0] = black_scholes_delta(path[0], T, strike, sigma, r)
    B[0] = opt_price - deltas[0] * path[0]

    to_maturity = np.zeros(1)
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
            deltas[i] = black_scholes_delta(path[i], T-i*dt, strike, sigma, r)
            B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]
            if i == num_steps-1:
                payoff = np.maximum(strike-path[i+1], 0)
                port_value = np.exp(r*dt)*B[i]+deltas[i]*path[i+1]
                error = payoff - port_value
                to_maturity[0] = 1


    return error, to_maturity
