import numpy as np
import pandas as pd
import time
from scipy import stats
from scipy.stats import norm
from Final.binomial_model import *
from Final.binomial_model_experiment import *
from Final.black_scholes import *
from Final.black_scholes_experiment import *
from Final.ISD_naive import *
from Final.ISD_naive_vf import *
from Final.ISD_naive_vf_experiment import *
from Final.LSM_algorithm import *
from Final.LSM_algorithm_experiment import *
from Final.simulation_of_paths import *

experiment_paths = pd.read_excel("/Users/marianilsson/Documents/Bachelorprojekt/experiment_paths.xlsx")
experiment_paths = experiment_paths.to_numpy()

boundary_prices = pd.read_excel("/Users/marianilsson/Documents/Bachelorprojekt/boundary_prices.xlsx")
boundary_prices = boundary_prices.iloc[:, 0].to_numpy()


def table_binomial_prices(steps, start_price, sigma, r, strike, T):
    """
    function to make table of binomial prices and corresponding execution time
    :param steps: list of steps
    :param start_price: initial price of the underlying stock
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity time
    :return: the table
    """
    table_binomial = pd.DataFrame()
    table_binomial["Number of steps"] = steps

    prices = np.zeros(len(steps))
    execution_time = np.zeros((len(steps), 50))
    for i in range(0, 50):
        for j in range(len(steps)):
            start = time.time()
            S, V = binomial_values(start_price, sigma, r, strike, T, steps[j])
            prices[j] = binomial_price(V)
            end = time.time()
            execution_time[j, i] = end - start

    table_binomial["Prices"] = prices
    table_binomial["Execution time (sec)"] = np.mean(execution_time, 1)

    return table_binomial


def table_1_LSM(r, sigma, S_0, strike, T):
    """
    function to make table of the price given different time steps and different paths
    :param r: risk-free rate
    :param sigma: volatility
    :param S_0: initial price of the underlying stock
    :param strike: the strike price
    :param T: maturity date
    :return: table
    """
    table = pd.DataFrame()
    table["M"] = [20, 50, 100, 20, 50, 100, 20, 50, 100, 20, 50, 100]
    table["N"] = [25000, 25000, 25000, 50000, 50000, 50000, 100000, 100000, 100000, 150000, 150000, 150000]

    prices = np.zeros((12, 50))
    execution_time = np.zeros((12, 50))
    for i in range(12):
        for j in range(50):
            start = time.time()
            sim_paths = monte_carlo_GBM(r, sigma, S_0, table["N"][i], table["M"][i], T)
            prices[i, j] = LSM_laguerre_price(sim_paths, r, strike, T)
            end = time.time()
            execution_time[i, j] = end - start

    table["LSM prices"] = np.mean(prices, 1)
    table["LSM StDev"] = np.std(prices, axis=1, ddof=1)
    table["LSM s.e."] = stats.sem(prices, axis=1)

    table["Average execution time"] = np.mean(execution_time, 1)

    return table, prices, execution_time


def table_2_LSM(boundary_prices):
    """
    table of prices using different methods and for different initial prices
    :param start_prices: list of initial prices of the stock
    :return: table and prices
    """
    start_prices = np.array([32,34,36,38,40,42,44,46,48])
    table = pd.DataFrame()
    table["Initial price"] = start_prices

    LSM_prices = np.zeros((len(start_prices), 100))
    for i in range(len(start_prices)-2):
        for j in range(100):
            sim_paths = monte_carlo_GBM(0.06, 0.2, start_prices[i], 100000, 50, 1)
            LSM_prices[i, j] = LSM_laguerre_price(sim_paths, 0.06, 40, 1)

    LSM_boundary_prices = np.zeros((len(start_prices), 100))
    # boundary_sim_paths = monte_carlo_GBM(0.06, 0.2, 36, 250000, 50, 1)
    # time_steps, boundary_prices = LSM_laguerre_boundary(boundary_sim_paths, 0.06, 40, 1)
    for i in range(len(start_prices)):
        for j in range(100):
            sim_paths_1 = monte_carlo_GBM(0.06, 0.2, start_prices[i], 100000, 50, 1)
            LSM_boundary_prices[i, j] = LSM_boundary_price(sim_paths_1, boundary_prices, 40, 0.02, 0.06)

    bin_prices = np.zeros(len(start_prices))
    for i in range(len(start_prices)):
        S, V = binomial_values(start_prices[i], 0.2, 0.06, 40, 1, 2000)
        bin_prices[i] = binomial_price(V)

    s = np.array(start_prices)
    d1 = (np.log(s / 40) + (0.06 + 0.5 * 0.2 ** 2) * 1) / (0.2 * np.sqrt(1))
    d2 = d1 - 0.2 * np.sqrt(1)
    BS_values = 40 * np.exp(-0.06 * 1) * norm.cdf(-d2) - s * norm.cdf(-d1)

    table["LSM-price"] = np.mean(LSM_prices, 1)
    table["LSM StDev"] = np.std(LSM_prices, axis=1, ddof=1)
    table["LSM s.e."] = stats.sem(LSM_prices, axis=1)

    table["Boundary price"] = np.mean(LSM_boundary_prices, 1)
    table["Boundary StDev"] = np.std(LSM_boundary_prices, axis=1, ddof=1)
    table["Boundary s.e."] = stats.sem(LSM_boundary_prices, axis=1)

    table["Difference"] = np.mean(LSM_prices, 1) - np.mean(LSM_boundary_prices, 1)
    table["Binomial price"] = bin_prices
    table["BS price"] = BS_values

    return table, LSM_prices, LSM_boundary_prices


def table_binomial_errors(experiment_paths):

    table = pd.DataFrame()

    binomial_errors = np.zeros(len(experiment_paths))
    # exercise boundary determined by the binomial model with 2000 steps
    time_steps, boundary_prices = binomial_boundary(36, 0.2, 0.06, 40, 1, 2000)
    for i in range(len(experiment_paths)):
        path = experiment_paths[i]
        binomial_errors[i] = binomial_delta_experiment_error(path, 0.2, 0.06, 40, 1, boundary_prices)

    table["Mean"] = np.mean(binomial_errors)
    table["Median"] = np.median(binomial_errors)
    table["Standard error"] = stats.sem(binomial_errors)
    table["Standard deviation"] = np.std(binomial_errors, ddof=1)

    return table


def table_ISD_M(boundary_prices):
    table = pd.DataFrame()

    table["S_0"] = [36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44,
                    36, 40, 44]
    table["M_0"] = [4, 4, 4, 8, 8, 8, 12, 12, 12, 4, 4, 4, 8, 8, 8, 12, 12, 12, 4, 4, 4, 8, 8, 8, 12, 12, 12]
    table["M"] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12]

    BM_price = np.zeros(27)
    BM_delta = np.zeros(27)
    for i in range(27):
        S, V = binomial_values(table["S_0"][i], 0.2, 0.06, 40, 1, 2000)
        BM_price[i] = binomial_price(V)
        BM_delta[i] = binomial_delta(S, V)
    table["BM_price"] = BM_price
    table["BM_delta"] = BM_delta

    ISD_price = np.zeros((27, 80))
    ISD_delta = np.zeros((27, 80))

    for i in range(27):
        for j in range(80):
            ISD_paths = ISD_paths_GBM(0.06, 0.2, table["S_0"][i], 100000, 50, 1, 5)
            X, Y = naive_vf_method(ISD_paths, 0.06, 40, 1, boundary_prices, table["M"][i])
            ISD_price[i, j], ISD_delta[i, j] = naive_vf_price_and_delta(X, Y, table["S_0"][i], table["M_0"][i])

    table["ISD_price"] = np.average(ISD_price, axis=1)
    table["StDev_price"] = np.std(ISD_price, axis=1, ddof=1)
    table["s.e. price"] = stats.sem(ISD_price, axis=1)

    table["ISD_delta"] = np.average(ISD_delta, axis=1)
    table["StDev_delta"] = np.std(ISD_delta, axis=1)
    table["s.e. delta"] = stats.sem(ISD_delta, axis=1)

    return table


def table_ISD_alpha(boundary_prices):
    table = pd.DataFrame()

    table["S_0"] = [36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44]
    table["alpha"] = [0.5, 0.5, 0.5, 5, 5, 5, 10, 10, 10, 25, 25, 25]

    BM_price = np.zeros(12)
    BM_delta = np.zeros(12)
    for i in range(12):
        S, V = binomial_values(table["S_0"][i], 0.2, 0.06, 40, 1, 2000)
        BM_price[i] = binomial_price(V)
        BM_delta[i] = binomial_delta(S, V)
    table["BM_price"] = BM_price
    table["BM_delta"] = BM_delta

    ISD_price = np.zeros((12, 50))
    ISD_delta = np.zeros((12, 50))

    for i in range(12):
        for j in range(50):
            ISD_paths = ISD_paths_GBM(0.06, 0.2, table["S_0"][i], 100000, 50, 1,
                                      table["alpha"][i])
            X, Y = naive_vf_method(ISD_paths, 0.06, 40, 1, boundary_prices, 8)
            ISD_price[i, j], ISD_delta[i, j] = naive_vf_price_and_delta(X, Y, table["S_0"][i], 8)

    table["ISD_price"] = np.average(ISD_price, axis=1)
    table["StDev_price"] = np.std(ISD_price, axis=1, ddof=1)
    table["s.e. price"] = stats.sem(ISD_price, axis=1)

    table["ISD_delta"] = np.average(ISD_delta, axis=1)
    table["StDev_delta"] = np.std(ISD_delta, axis=1, ddof=1)
    table["s.e. delta"] = stats.sem(ISD_delta, axis=1)

    return table


def table_LSM_appendix(total_number):
    table = pd.DataFrame()

    table["S_0"] = [36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44]
    table["sigma"] = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4]
    table["T"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    BM_price = np.zeros(18)
    for i in range(18):
        S, V = binomial_values(table["S_0"][i], table["sigma"][i], 0.06, 40, table["T"][i], 2000)
        BM_price[i] = binomial_price(V)
    table["BM_price"] = BM_price

    LSM_prices = np.zeros((16, total_number))
    for i in range(16):
        for j in range(total_number):
            sim_paths = monte_carlo_GBM(0.06, table["sigma"][i], table["S_0"][i], 100000, 50,
                                        table["T"][i])
            LSM_prices[i, j] = LSM_price(sim_paths, 0.06, 40, table["T"][i])

    table["LSM_prices"] = np.average(LSM_prices, axis=1)
    table["StDev_price"] = np.std(LSM_prices, axis=1, ddof=1)
    table["s.e. price"] = stats.sem(LSM_prices, axis=1)

    return table, LSM_prices


def table_ISD_paths(boundary_prices):
    table = pd.DataFrame()

    table["S_0"] = [36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44]
    table["N"] = [50000, 50000, 50000, 100000, 100000, 100000, 150000, 150000, 150000, 200000, 200000, 200000]

    BM_price = np.zeros(12)
    BM_delta = np.zeros(12)
    for i in range(12):
        S, V = binomial_values(table["S_0"][i], 0.2, 0.06, 40, 1, 2000)
        BM_price[i] = binomial_price(V)
        BM_delta[i] = binomial_delta(S, V)
    table["BM_price"] = BM_price
    table["BM_delta"] = BM_delta

    ISD_prices = np.zeros((12, 50))
    ISD_deltas = np.zeros((12, 50))

    for i in range(12):
        for j in range(50):
            ISD_paths = ISD_paths_GBM(0.06, 0.2, table["S_0"][i], table["N"][i], 50, 1, 5)
            X, Y = naive_vf_method(ISD_paths, 0.06, 40, 1, boundary_prices, 8)
            ISD_prices[i, j], ISD_deltas[i, j] = naive_vf_price_and_delta(X, Y, table["S_0"][i], 8)

    table["ISD_price"] = np.average(ISD_prices, axis=1)
    table["StDev_price"] = np.std(ISD_prices, axis=1, ddof=1)
    table["s.e. price"] = stats.sem(ISD_prices, axis=1)
    table["Price_diff"] = BM_price - np.average(ISD_prices, axis=1)

    table["ISD_delta"] = np.average(ISD_deltas, axis=1)
    table["StDev_delta"] = np.std(ISD_deltas, axis=1, ddof=1)
    table["s.e. delta"] = stats.sem(ISD_deltas, axis=1)
    table["Delta_diff"] = BM_delta - np.average(ISD_deltas, axis=1)

    return table, ISD_prices, ISD_deltas


def table_ISD_steps(boundary_prices):
    table = pd.DataFrame()

    table["S_0"] = [36, 40, 44, 36, 40, 44, 36, 40, 44]
    table["Steps"] = [20, 20, 20, 50, 50, 50, 100, 100, 100]

    BM_price = np.zeros(9)
    BM_delta = np.zeros(9)
    for i in range(9):
        S, V = binomial_values(table["S_0"][i], 0.2, 0.06, 40, 1, 2000)
        BM_price[i] = binomial_price(V)
        BM_delta[i] = binomial_delta(S, V)
    table["BM_price"] = BM_price
    table["BM_delta"] = BM_delta

    ISD_prices = np.zeros((9, 50))
    ISD_deltas = np.zeros((9, 50))

    sim_paths0 = monte_carlo_GBM(0.06, 0.2, 36, 250000, 20, 1)
    time_steps0, boundary_prices0 = LSM_boundary(sim_paths0, 0.06, 40, 1)

    sim_paths1 = monte_carlo_GBM(0.06, 0.2, 36, 250000, 100, 1)
    time_steps1, boundary_prices1 = LSM_boundary(sim_paths1, 0.06, 40, 1)

    for i in range(9):
        for j in range(50):
            if i < 3:
                boundary_prices_1 = boundary_prices0
            if i > 5:
                boundary_prices_1 = boundary_prices1
            if 2 < i < 6:
                boundary_prices_1 = boundary_prices
            ISD_paths = ISD_paths_GBM(0.06, 0.2, table["S_0"][i], 100000, table["Steps"][i],
                                      1, 5)
            X, Y = naive_vf_method(ISD_paths, 0.06, 40, 1, boundary_prices_1, 8)
            ISD_prices[i, j], ISD_deltas[i, j] = naive_vf_price_and_delta(X, Y, table["S_0"][i], 8)

    table["ISD_price"] = np.average(ISD_prices, axis=1)
    table["StDev_price"] = np.std(ISD_prices, axis=1, ddof=1)
    table["s.e. price"] = stats.sem(ISD_prices, axis=1)
    table["Price_diff"] = BM_price - np.average(ISD_prices, axis=1)

    table["ISD_delta"] = np.average(ISD_deltas, axis=1)
    table["StDev_delta"] = np.std(ISD_deltas, axis=1, ddof=1)
    table["s.e. delta"] = stats.sem(ISD_deltas, axis=1)
    table["Delta_diff"] = BM_delta - np.average(ISD_deltas, axis=1)

    return table, ISD_prices, ISD_deltas


def table_ISD_appendix(boundary_prices):
    table = pd.DataFrame()

    table["S_0"] = [36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44, 36, 40, 44]
    table["sigma"] = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4]
    table["T"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    BM_price = np.zeros(18)
    BM_delta = np.zeros(18)
    for i in range(18):
        S, V = binomial_values(table["S_0"][i], table["sigma"][i], 0.06, 40, table["T"][i], 2000)
        BM_price[i] = binomial_price(V)
        BM_delta[i] = binomial_delta(S, V)
    table["BM_price"] = BM_price
    table["BM_delta"] = BM_delta

    ISD_prices = np.zeros((18, 50))
    ISD_deltas = np.zeros((18, 50))

    for i in range(9):
        for j in range(50):
            ISD_paths = ISD_paths_GBM(0.06, table["sigma"][i], table["S_0"][i], 100000, 50,
                                      table["T"][i], 5)
            X, Y = naive_vf_method(ISD_paths, 0.06, 40, table["T"][i], boundary_prices, 8)
            ISD_prices[i, j], ISD_deltas[i, j] = naive_vf_price_and_delta(X, Y, table["S_0"][i], 8)

    sim_paths = monte_carlo_GBM(0.06, 0.2, 36, 250000, 50, 2)
    time_steps, boundary_prices1 = LSM_boundary(sim_paths, 0.06, 40, 2)

    for i in range(9, 18):
        for j in range(50):
            ISD_paths = ISD_paths_GBM(0.06, table["sigma"][i], table["S_0"][i], 100000, 50,
                                      table["T"][i], 5)
            X, Y = naive_vf_method(ISD_paths, 0.06, 40, table["T"][i], boundary_prices1, 8)
            ISD_prices[i, j], ISD_deltas[i, j] = naive_vf_price_and_delta(X, Y, table["S_0"][i], 8)

    table["ISD_price"] = np.average(ISD_prices, axis=1)
    table["StDev_price"] = np.std(ISD_prices, axis=1, ddof=1)
    table["s.e. price"] = stats.sem(ISD_prices, axis=1)
    table["Price_diff"] = BM_price - np.average(ISD_prices, axis=1)

    table["ISD_delta"] = np.average(ISD_deltas, axis=1)
    table["StDev_delta"] = np.std(ISD_deltas, axis=1, ddof=1)
    table["s.e. delta"] = stats.sem(ISD_deltas, axis=1)
    table["Delta_diff"] = BM_delta - np.average(ISD_deltas, axis=1)

    return table, ISD_prices, ISD_deltas


def table_LSM_basis():
    table = pd.DataFrame()

    table["Basis functions"] = ["Weighted Laguerre", "Polynomial (3. deg)", "Hermite", "Legendre"]

    LSM_values = np.zeros(4)
    sim_paths = monte_carlo_GBM(0.06,0.2,36,100000,50,1)

    LSM_values[0] = LSM_price(sim_paths,0.06,40,1)
    C_poly = LSM_algorithm_poly(sim_paths,0.06,40,1,3)
    LSM_values[1] = LSM_price_1(C_poly,0.02,0.06,1)
    C_hermite = LSM_algorithm_hermite(sim_paths,0.06,40,1)
    LSM_values[2] = LSM_price_1(C_hermite,0.02,0.06,1)
    C_legendre = LSM_algorithm_legendre(sim_paths,0.06,40,1)
    LSM_values[3] = LSM_price_1(C_legendre,0.02,0.06,1)

    table["LSM prices"] = LSM_values

    return table
