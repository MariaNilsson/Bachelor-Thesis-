import numpy as np
from Final.simulation_of_paths import *
from Final.LSM_algorithm import *


def LSM_delta_experiment_error(path, opt_price, r, sigma, T, strike, boundary_prices, num_paths):
    """
    Function to calculate the hedging error using the LSM-algorithm.
    :param path: path of stock prices
    :param price: the price of the option at time 0
    :param price_epsilon: the price of the option at time 0 with epsilon added to the initial price of the stock
    :param r: risk-free rate
    :param sigma: volatility
    :param T: time to maturity
    :param strike: strike price
    :param boundary_prices: prices of the early exercise boundary determined by the binomial model
    :param num_paths: number of paths used in the LSM-algorithm
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

    # for time 0
    sim_paths = monte_carlo_GBM(r, sigma, path[0], num_paths, len(path)-1, T)
    sim_paths_epsilon = monte_carlo_GBM(r, sigma, path[0] * 1.03, num_steps, len(path)-1, T)

    price = LSM_boundary_price(sim_paths, boundary_prices, strike, dt, r)
    price_epsilon = LSM_boundary_price(sim_paths_epsilon, boundary_prices, strike, dt, r)

    deltas[0] = (price_epsilon - price) / (path[0]*0.03)
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
            epsilon = 0.03*path[i]
            # simulated paths with initial price path[i]
            sim_paths = monte_carlo_GBM_dt(r, sigma, path[i], num_paths, dt, num_steps - i)
            # simulated paths with initial price path[i] + epsilon
            sim_paths_epsilon = monte_carlo_GBM_dt(r, sigma, path[i] + epsilon, num_paths, dt, num_steps - i)

            # price for sim_path
            price = LSM_boundary_price(sim_paths, boundary_prices[i:], strike, dt, r)
            # price for sim_paths_epsilon
            price_epsilon = LSM_boundary_price(sim_paths_epsilon, boundary_prices[i:], strike, dt, r)

            # deltas
            deltas[i] = (price_epsilon - price) / epsilon
            if deltas[i] < -1:
                deltas[i] = -1
            if deltas[i] > 0:
                deltas[i] = 0

            B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]

            if i == num_steps-1:
                payoff = np.maximum(strike-path[i+1], 0)
                port_value = np.exp(r*dt)*B[i]+deltas[i]*path[i+1]
                error = payoff - port_value

    return error

