import numpy as np
from Final.simulation_of_paths import *
from Final.ISD_naive_vf import *
from Final.ISD_naive import *
from Final.black_scholes import *


def ISD_delta_experiment_error_1(path, opt_price, sigma, r, strike, T, boundary_prices, alpha, deg, M_0):
    """
    Function to calculate the hedging error using the ISD-naive-value-function method.
    :param path: path of stock prices
    :param opt_price: time 0 price of the option
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: strike price
    :param T: time to maturity
    :param boundary_prices: prices of the early exercise boundary determined by the binomial model
    :param alpha: bandwidth of initial prices
    :param deg: degree of polynomial at time t_1
    :param M_0: degree of polynomial at time t_0
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
    ISD_paths_0 = ISD_paths_GBM_dt(r, sigma, path[0], 100000, num_steps, dt, alpha)
    X_0, Y_0 = naive_vf_method_dt(ISD_paths_0, r, strike, dt, num_steps, boundary_prices, deg)
    deltas[0] = naive_vf_delta(X_0, Y_0, path[0], M_0)
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
            if i == num_steps - 1:
                deltas[i] = black_scholes_delta(path[i], dt, strike, sigma, r)
                B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]
                payoff = np.maximum(strike - path[i + 1], 0)
                port_value = np.exp(r * dt) * B[i] + deltas[i] * path[i + 1]
                error = payoff - port_value
            else:
                ISD_paths = ISD_paths_GBM_dt(r, sigma, path[i], 100000, num_steps - i, dt, alpha)
                X, Y = naive_vf_method_dt(ISD_paths, r, strike, dt, num_steps - i, boundary_prices[i:], deg)
                deltas[i] = naive_vf_delta(X, Y, path[i], M_0)
                if deltas[i] < -1:
                    deltas[i] = -1
                if deltas[i] > 0:
                    deltas[i] = 0
                B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]

    return error


def ISD_delta_functions_coef(ISD_paths, x_0, M_0, r, T, strike, boundary_prices, deg):
    """
    function to determine coefficients for each value function
    :param ISD_paths: matrix with ISD paths
    :param x_0: initial price
    :param M_0: degree of polynomial at time t_0
    :param r: risk-free rate
    :param T: maturity date
    :param strike: strike price
    :param boundary_prices: vector of boundary prices
    :param deg: degree of polynomial at time t_1
    :return: coefficients
    """
    # total number of steps
    num_steps = len(ISD_paths[0]) - 1
    dt = T / num_steps

    coefficients = np.zeros((M_0 + 1, num_steps))

    for i in range(num_steps):
        if num_steps-i == 1:
            X, Z = naive_method_dt(ISD_paths[:, i:], strike, dt, num_steps-i, r, boundary_prices[i:])
            coefficients[:, i] = naive_method_coef(ISD_paths[:, i:], Z, x_0, M_0)
        else:
            X, Y = naive_vf_method_dt(ISD_paths[:, i:], r, strike, dt, num_steps - i, boundary_prices[i:], deg)
            coefficients[:, i] = naive_vf_coef(X, Y, x_0, M_0)

    return coefficients


def ISD_delta_function(coefficients, S, x_0):
    """
    function to return delta given coefficients in price function
    :param coefficients: coefficients of the price function
    :param S: price of the underlying stock
    :param x_0: initial price
    :return: delta
    """
    delta = 0
    for i in range(1, len(coefficients)):
        delta = delta + coefficients[i]*i*(S-x_0)**(i-1)

    return delta


def ISD_delta_experiment_error_2(path, opt_price, coefficients, r, sigma, x_0, strike, T, boundary_prices):
    # number of total steps
    num_steps = len(path) - 1
    # length of each time step
    dt = T / num_steps

    # defining vector of zeroes to store delta for each step
    deltas = np.zeros(num_steps)
    # defining vector of zeroes to store the risk-free investment
    B = np.zeros(num_steps)

    # for time t=0
    deltas[0] = ISD_delta_function(coefficients[:, 0], path[0], x_0)
    B[0] = opt_price - deltas[0] * path[0]

    for i in range(1, num_steps):
        # if the option is exercised
        if path[i] <= boundary_prices[i-0]:
            payoff = np.maximum(strike - path[i], 0)
            port_value = np.exp(r * dt) * B[i - 1] + deltas[i - 1] * path[i]
            error = payoff - port_value
            error = error * np.exp(r * (num_steps - i) * dt)
            break

        # if the option has not been exercised
        else:
            if i == num_steps - 1:
                deltas[i] = black_scholes_delta(path[i], dt, strike, sigma, r)
                B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]
                payoff = np.maximum(strike - path[i + 1], 0)
                port_value = np.exp(r * dt) * B[i] + deltas[i] * path[i + 1]
                error = payoff - port_value
            else:
                deltas[i] = ISD_delta_function(coefficients[:, i], path[i], x_0)
                if deltas[i] < -1:
                    deltas[i] = -1
                if deltas[i] > 0:
                    deltas[i] = 0
                B[i] = np.exp(r * dt) * B[i - 1] - (deltas[i] - deltas[i - 1]) * path[i]

    return error
