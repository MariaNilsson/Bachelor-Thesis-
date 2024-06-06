from LSM.LSM_algorithm import *
import statsmodels.api as sm
import numpy as np


def naive_method(ISD_paths, r, strike, T, boundary_prices):
    """
    function which returns initial prices and corresponding discounted cash-flows using the naive method
    :param ISD_paths: matrix with ISD simulated paths
    :param r: risk-free rate
    :param strike: strike price
    :param T: maturity date
    :param boundary_prices: vector of boundary prices
    :return: initial prices and discounted cash-flows
    """
    # get matrix containing cash-flows for each ISD-path following the LSM boundary
    ISD_cashflows = np.zeros([len(ISD_paths), (len(ISD_paths[0])-1)])
    for i in range(0, len(ISD_paths)):
        for j in range(0, len(ISD_paths[0])-1):
            if ISD_paths[i, j+1] <= boundary_prices[j]:
                ISD_cashflows[i, j] = strike-ISD_paths[i, j+1]
                break
            else:
                ISD_cashflows[i, j] = 0

    # discount all cash-flows back to time t=0
    dt = T/(len(ISD_paths[0])-1)
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)

    # discounted payoffs
    Z = np.array(ISD_cashflows.dot(disc_vector))

    return ISD_paths[:, 0], Z


def naive_method_dt(ISD_paths, strike, dt, total_steps, r, boundary_prices):
    """
    function which returns initial prices and corresponding discounted cash-flows using the naive method
    :param ISD_paths: matrix with ISD simulated paths
    :param strike: strike price
    :param dt: maturity date
    :param total_steps: total number of steps
    :param r: risk-free rate
    :param boundary_prices: vector of boundary prices
    :return: initial prices and discounted cash-flows
    """
    # get matrix containing cash-flows for each ISD-path following the LSM boundary
    ISD_cashflows = np.zeros([len(ISD_paths), (len(ISD_paths[0])-1)])
    for i in range(0, len(ISD_paths)):
        for j in range(0, len(ISD_paths[0])-1):
            if ISD_paths[i, j+1] <= boundary_prices[j]:
                ISD_cashflows[i, j] = strike-ISD_paths[i, j+1]
                break
            else:
                ISD_cashflows[i, j] = 0

    # discount all cash-flows back to time t=0
    t = np.arange(dt, dt*total_steps+0.001, dt)
    disc_vector = np.exp(-r * t)

    # discounted payoffs
    Z = np.array(ISD_cashflows.dot(disc_vector))

    return ISD_paths[:, 0], Z


def naive_method_price_and_delta(ISD_paths, Z, x_0, M_0):
    """
    function to determine the price and delta of the option
    :param ISD_paths: matrix containing ISD paths
    :param Z: discounted cash-flows following an estimated optimal exercise strategy
    :param x_0: initial price
    :param M_0: degree of the polynomial to determine the price function
    :return: price and delta
    """
    # model matrix
    model_matrix = np.zeros([len(Z), M_0 + 1])
    for i in range(0, M_0 + 1):
        model_matrix[:, i] = (ISD_paths[:, 0] - x_0) ** i

    # OLS regression
    model = sm.OLS(Z, model_matrix)
    results = model.fit()
    params = results.params

    price = params[0]
    delta = params[1]

    return price, delta


def naive_method_coef(ISD_paths, Z, x_0, M_0):
    """
    function to determine the price and delta of the option
    :param ISD_paths: matrix containing ISD paths
    :param Z: discounted cash-flows following an estimated optimal exercise strategy
    :param x_0: initial price
    :param M_0: degree of the polynomial to determine the price function
    :return: price and delta
    """
    # model matrix
    model_matrix = np.zeros([len(Z), M_0 + 1])
    for i in range(0, M_0 + 1):
        model_matrix[:, i] = (ISD_paths[:, 0] - x_0) ** i

    # OLS regression
    model = sm.OLS(Z, model_matrix)
    results = model.fit()
    params = results.params

    return params
