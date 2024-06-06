import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
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
import seaborn as sns


experiment_paths = pd.read_excel("/Users/marianilsson/Documents/Bachelorprojekt/experiment_paths.xlsx")
experiment_paths = experiment_paths.to_numpy()

boundary_prices = pd.read_excel("/Users/marianilsson/Documents/Bachelorprojekt/boundary_prices.xlsx")
boundary_prices = boundary_prices.iloc[:, 0].to_numpy()


def plot_european_value(t, T, K, r, sigma):
    """
    plot of the pricing function and payoff function
    :param t: current time
    :param T: maturity time
    :param K: strike price
    :param r: risk-free rate
    :param sigma: volatility
    :return: plot
    """
    s = np.arange(20, 60, 1)
    call_values = (s * norm.cdf(1/(sigma*np.sqrt(T-t))*(np.log(s/K)+(r+1/2*sigma**2)*(T-t))) - np.exp(-r*(T-t)) * K
              * norm.cdf(1/(sigma*np.sqrt(T-t))*(np.log(s/K)+(r+1/2*sigma**2)*(T-t)) - sigma*np.sqrt(T-t)))

    put_values = call_values + K*np.exp(-r*(T-t))-s

    plt.plot(s, call_values)
    plt.plot(s, put_values)
    plt.legend(["European call", "European put"])
    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Value of the option")

    return plt.show()


def plot_binomial_lattice(start_price, steps):
    """
    plot binomial lattice
    :param start_price: start price of the stock
    :param steps: number of steps
    :return: the plot
    """
    S, V = binomial_values(start_price, 0.2, 0.06, 40, 1, steps)
    dt = 1/steps
    for i in range(steps+1):
        X = np.arange(i*dt, 1+0.001, dt)
        Y = S[-(i+1), i:]
        plt.scatter(X, Y, s=10, color='#1f77b4')
        plt.plot(X,Y, color='#1f77b4', linewidth=1)
        Y_1 = np.zeros(steps+1-i)
        for j in range(i, steps+1):
            Y_1[j-i] = max(S[steps-j+i:,j])
        plt.plot(X, Y_1, color='#1f77b4', linewidth=1)

    plt.ylabel("Price of the underlying stock")
    plt.xlabel("Time")

    return plt.show()


def plot_simulated_paths(start_price, num_paths, num_steps):
    """
    plot example of simulated paths using GBM
    :param start_price: the start price of the underlying stock
    :param num_paths: the number of paths
    :param num_steps: the number of steps pr. year
    :return: the ploy
    """
    paths = monte_carlo_GBM(0.06, 0.2, start_price, num_paths, num_steps, 1)
    dt = 1/num_steps
    X = np.arange(0, 1+0.001, dt)
    for i in range(len(paths)):
        plt.plot(X, paths[i])

    plt.xlabel("Time")
    plt.ylabel("Price of the underlying stock")

    return plt.show()


def plot_binomial_boundary(steps, start_price, sigma, r, strike, T):
    """
    plot early exercise boundary for different number of time steps using the binomial model
    :param steps: list of number of steps
    :param start_price: the initial price of the underlying stock
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity time
    :return: plot
    """
    for i in range(len(steps)):
        X, Y = binomial_boundary(start_price, sigma, r, strike, T, steps[i])
        plt.plot(X, Y, linewidth=0.8, color='#9467bd')

    plt.xlabel("Time")
    plt.ylabel("Price of the underlying stock")
    legend = ['M=' + str(steps[0])]
    for i in range(1, len(steps)):
        legend.append('M=' + str(steps[i]))
    plt.legend(legend)

    return plt.show()


def plot_binomial_convergence(step_start, step_interval, step_end):

    steps = np.arange(step_start, step_end+step_interval, step_interval)

    prices = np.zeros(len(steps))
    for i in range(len(steps)):
        S, V = binomial_values(36, 0.2, 0.06, 40, 1, steps[i])
        prices[i] = binomial_price(V)

    plt.plot(steps, prices, linewidth=0.75)
    plt.xlabel("Number of steps")
    plt.ylabel("Value of the option")

    return plt.show()


def plot_LSM_boundary_1(num_paths, num_steps):
    """
    plot exercise boundary using LSM with different choices of number of paths and steps
    :param num_paths: list with number of paths
    :param num_steps: list with number of steps
    :return: plot
    """
    for i in range(len(num_paths)):
        sim_paths = monte_carlo_GBM(0.06, 0.2, 36, num_paths[i], num_steps[i], 1)
        time_steps, boundary_prices = LSM_laguerre_boundary(sim_paths, 0.06, 40, 1)
        plt.plot(time_steps, boundary_prices, marker='o', markersize=1.5, linewidth=0.8)

    plt.grid(color='lightgrey')
    plt.xlabel("Time")
    plt.ylabel("Price of the underlying stock")

    legend = ['N=' + str(num_paths[0]) + ', M=' + str(num_steps[0])]
    for i in range(1, len(num_paths)):
        legend.append('N=' + str(num_paths[i]) + ', M=' + str(num_steps[i]))
    plt.legend(legend)

    return plt.show()


def plot_LSM_boundary_2(start_prices):
    """
    plot exercise boundary using LSM for different initial prices
    :param start_prices: list of different start prices of the stock
    :return: plot
    """
    for i in range(len(start_prices)):
        sim_paths = monte_carlo_GBM(0.06, 0.2, start_prices[i], 100000, 100, 1)
        time_steps, boundary_prices = LSM_laguerre_boundary(sim_paths, 0.06, 40, 1)
        plt.plot(time_steps, boundary_prices, marker='o', markersize=2)

    plt.grid(color='lightgrey')
    plt.xlabel("Time")
    plt.ylabel("Price of the underlying stock")

    legend = ['Initial price: ' + str(start_prices[0])]
    for i in range(1, len(start_prices)):
        legend.append('Initial price: ' + str(start_prices[i]))
    plt.legend(legend)

    return plt.show()


def plot_LSM_bin_boundary(bin_steps, LSM_num_paths, LSM_steps):
    """
    plot the exercise boundary approximated by the binomial model and the LSM-algorithm
    :param bin_steps: steps used in the binomial model
    :param LSM_num_paths: number of paths used in the LSM-algorithm
    :param LSM_steps: number of steps used in the LSM-algorithm
    :return: plot
    """
    X, Y = binomial_boundary(36, 0.2, 0.06, 40, 1, bin_steps)
    plt.plot(X, Y, linewidth=0.8)

    sim_paths = monte_carlo_GBM(0.06, 0.2, 36, LSM_num_paths, LSM_steps, 1)
    time_steps, boundary_prices = LSM_laguerre_boundary(sim_paths, 0.06, 40, 1)
    plt.plot(time_steps, boundary_prices, marker='o', markersize=2)

    plt.grid(color='lightgrey')
    plt.xlabel("Time")
    plt.ylabel("Price of the underlying stock")
    plt.legend(["Binomial", "LSM"])

    return plt.show()


def plot_value_fun(bin_steps):
    s = np.arange(20, 60, 1)
    bin_values = np.zeros(len(s))
    LSM_values = np.zeros(len(s))
    for i in range(len(s)):
        S, V = binomial_values_dt(s[i], 0.2, 0.06,40,0.0005, bin_steps)
        bin_values[i] = binomial_price(V)
        sim_paths = monte_carlo_GBM_dt(0.06,0.2,s[i],100000,0.02,50)
        LSM_values[i] = LSM_boundary_price(sim_paths,boundary_prices,40,0.02,0.06)

    plt.plot(s, bin_values, linewidth=3)
    plt.plot(s, LSM_values, linewidth=1)
    plt.scatter(s, LSM_values, color='red', s=1)

    return plt.show()


def plot_LSM_convergence():
    paths = np.arange(5000, 1000000+1, 5000)
    LSM_values = np.zeros(len(paths))
    for i in range(len(paths)):
        sim_paths = monte_carlo_GBM(0.06, 0.2, 36, paths[i], 50, 1)
        LSM_values[i] = LSM_price(sim_paths, 0.06, 40, 1)

    plt.plot(paths, LSM_values)

    return plt.show()


def plot_initial_values(N, x_0, alpha):
    """
    Plot histogram of initial values of the underlying stock
    :param N: number of initial values
    :param x_0: initial price
    :param alpha: bandwidth
    :return: histogram
    """
    X = np.zeros(N)
    U = np.random.uniform(0, 1, N)
    for i in range(N):
        X[i] = x_0 + alpha * 2 * np.sin(np.arcsin(2 * U[i] - 1) / 3)

    plt.hist(X, weights=np.ones(len(X)) / len(X), bins=40, edgecolor='white')
    plt.xlabel("Initial price of the underlying stock")
    plt.ylabel("Density")

    return plt.show()


def plot_paths(S, T):
    dt = T/(len(S[0])-1)
    X = np.arange(0, T+dt, dt)
    for i in range(len(S)):
        plt.plot(X, S[i])
        plt.xlabel("Time")
        plt.ylabel("Price of the underlying stock")

    return plt.show()


def plot_ISD_data(boundary_prices):
    ISD_paths_low = ISD_paths_GBM(0.06, 0.2, 36, 100000, 50, 1, 0.5)
    ISD_paths_middle = ISD_paths_GBM(0.06, 0.2, 36, 100000, 50, 1, 5)
    ISD_paths_high = ISD_paths_GBM(0.06, 0.2, 36, 100000, 50, 1, 25)

    prices1, discounted_cashflows1 = naive_method(ISD_paths_low, 0.06, 40, 1, boundary_prices)
    plt.subplot(2, 3, 1)
    plt.scatter(prices1, discounted_cashflows1, s=0.2)
    plt.xlabel("(a)")

    prices2, discounted_cashflows2 = naive_method(ISD_paths_middle, 0.06, 40, 1, boundary_prices)
    plt.subplot(2, 3, 2)
    plt.scatter(prices2, discounted_cashflows2, s=0.2)
    plt.xlabel("(b)")

    prices3, discounted_cashflows3 = naive_method(ISD_paths_high, 0.06, 40, 1, boundary_prices)
    plt.subplot(2, 3, 3)
    plt.scatter(prices3, discounted_cashflows3, s=0.2)
    plt.xlabel("(c)")

    prices4, discounted_cashflows4 = naive_vf_method(ISD_paths_low, 0.06, 40, 1, boundary_prices, 8)
    plt.subplot(2, 3, 4)
    plt.scatter(prices4, discounted_cashflows4, s=0.2)
    plt.xlabel("(d)")
    plt.ylim(0, 10)

    prices5, discounted_cashflows5 = naive_vf_method(ISD_paths_middle, 0.06, 40, 1, boundary_prices, 8)
    plt.subplot(2, 3, 5)
    plt.scatter(prices5, discounted_cashflows5, s=0.2)
    plt.xlabel("(e)")

    prices6, discounted_cashflows6 = naive_vf_method(ISD_paths_high, 0.06, 40, 1, boundary_prices, 8)
    plt.subplot(2, 3, 6)
    plt.scatter(prices6, discounted_cashflows6, s=0.2)
    plt.xlabel("(f)")

    return plt.show()


def plot_binomial_delta_fun(steps):
    """
    plot the binomial function for different choices of time steps
    :param steps: list of steps
    :return: plot
    """
    prices = np.arange(15, 70, 0.5)
    for i in range(len(steps)):
        deltas = np.zeros(len(prices))
        for j in range(len(prices)):
            S, V = binomial_values(prices[j], 0.2, 0.06, 40, 1, steps[i])
            deltas[j] = binomial_delta(S, V)

        plt.scatter(prices, deltas, s=5)

    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Delta")

    legend = ['Steps: ' + str(steps[0])]
    for i in range(1, len(steps)):
        legend.append('Steps: ' + str(steps[i]))
    plt.legend(legend)

    return plt.show()


def plot_binomial_delta_errors(experiment_paths, boundary_prices, opt_price):
    """
    given paths for the experiment the function returns the plot of the distribution
    :param experiment_paths: simulated paths for the experiment
    :return: plot
    """
    binomial_errors = np.zeros(len(experiment_paths))
    for i in range(len(experiment_paths)):
        path = experiment_paths[i]
        binomial_errors[i] = binomial_delta_experiment_error(path, opt_price, 0.2, 0.06, 40, 1,
                                                             boundary_prices)

    plt.hist(binomial_errors, density=True, bins=int((max(binomial_errors)-min(binomial_errors))/0.14), edgecolor='white')
    plt.ylabel("Density")
    plt.xlabel("Hedging-error")
    plt.xlim(-3.6, 3.6)
    plt.ylim(0, 1.7)

    return plt.show()


def plot_LSM_delta_fun(bin_steps, boundary_prices):
    prices = np.arange(15, 70, 0.5)
    bin_deltas = np.zeros(len(prices))
    LSM_deltas_1 = np.zeros(len(prices))
    LSM_deltas_2 = np.zeros(len(prices))
    LSM_deltas_3 = np.zeros(len(prices))
    for i in range(len(prices)):
        S, V = binomial_values(prices[i], 0.2, 0.06, 40, 1, bin_steps)
        bin_deltas[i] = binomial_delta(S, V)
        LSM_deltas_1[i] = LSM_laguerre_delta(boundary_prices, prices[i], 0.5 * prices[i], 0.06, 0.2, 40,
                                             1, 100000, 50)
        LSM_deltas_2[i] = LSM_laguerre_delta(boundary_prices, prices[i], 0.03 * prices[i], 0.06, 0.2, 40,
                                             1, 100000, 50)
        LSM_deltas_3[i] = LSM_laguerre_delta(boundary_prices, prices[i], 0.01 * prices[i], 0.06, 0.2, 40,
                                             1, 100000, 50)

    plt.plot(prices, bin_deltas, linewidth=1.5)
    plt.scatter(prices, LSM_deltas_1, s=8, color='#ff7f0e')
    plt.scatter(prices, LSM_deltas_2, s=8, color='#2ca02c')
    plt.scatter(prices, LSM_deltas_3, s=8, color='#e377c2')

    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Delta")
    plt.legend(["Binomial", "epsilon=0.5*S", "epsilon=0.03*S", "epsilon=0.01*S"])

    return plt.show()


def plot_LSM_delta_fun_1(bin_steps, boundary_prices, time_to_maturity):
    prices = np.arange(20, 70, 0.5)
    bin_deltas = np.zeros(len(prices))
    bin_dt = time_to_maturity/bin_steps

    LSM_deltas_1 = np.zeros(len(prices))
    LSM_deltas_2 = np.zeros(len(prices))
    LSM_deltas_3 = np.zeros(len(prices))
    for i in range(len(prices)):
        S, V = binomial_values_dt(prices[i],0.2,0.06,40, bin_dt, bin_steps)
        bin_deltas[i] = binomial_delta(S, V)

        LSM_deltas_1[i] = LSM_delta_dt(boundary_prices[45:], prices[i], 0.03 * prices[i], 0.06, 0.2, 40, 0.02, 100000, 5)
        LSM_deltas_2[i] = LSM_delta_dt(boundary_prices[45:], prices[i], 0.03 * prices[i], 0.06, 0.2, 40, 0.02, 50000, 5)
        LSM_deltas_3[i] = LSM_delta_dt(boundary_prices[45:], prices[i], 0.03 * prices[i], 0.06, 0.2, 40, 0.02, 25000, 5)

    plt.plot(prices, bin_deltas, linewidth=1.5)
    plt.scatter(prices, LSM_deltas_1, s=6, color='#ff7f0e')
    plt.scatter(prices, LSM_deltas_2, s=6, color='#2ca02c')
    plt.scatter(prices, LSM_deltas_3, s=6, color='#e377c2')

    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Delta")
    plt.legend(["Binomial", "Paths: 100000", "Paths: 50000", "Paths: 25000"])

    return plt.show()


def plot_LSM_delta_errors(experiment_paths, boundary_prices, num_paths, opt_price):
    """
    given paths for the experiment the function returns the plot of the distribution
    :param experiment_paths: simulated paths for the experiment
    :return: plot
    """
    LSM_errors = np.zeros(len(experiment_paths))

    for i in range(400, 500):
        path = experiment_paths[i]
        LSM_errors[i] = LSM_delta_experiment_error(path, opt_price, 0.06, 0.2, 1, 40, boundary_prices,
                                                   num_paths)

    plt.hist(LSM_errors, density=True, bins=int((max(LSM_errors)-min(LSM_errors))/0.14), edgecolor='white')
    plt.ylabel("Density")
    plt.xlabel("Hedging-error")
    plt.xlim(-3.6, 3.6)
    plt.ylim(0, 1.7)

    return plt.show()


def plot_BS_delta_fun(time_to_maturity):
    prices = np.arange(15, 70, 0.5)
    bin_deltas = np.zeros(len(prices))
    BS_deltas = np.zeros(len(prices))
    for i in range(len(prices)):
        S, V = binomial_values_dt(prices[i], 0.2, 0.06, 40, time_to_maturity/50, 50)
        bin_deltas[i] = binomial_delta(S, V)
        BS_deltas[i] = norm.cdf(1 / (0.2 * np.sqrt(time_to_maturity)) *
                                (np.log(prices[i] / 40) + (0.06 + 0.5 * 0.2 ** 2) * time_to_maturity)) - 1

    plt.plot(prices, bin_deltas)
    plt.plot(prices, BS_deltas)

    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Delta")
    plt.legend(["Binomial", "Black-Scholes"])

    return plt.show()


def plot_BS_delta_errors(experiment_paths, boundary_prices, opt_price):

    BS_errors = np.zeros(len(experiment_paths))

    for i in range(len(experiment_paths)):
        path = experiment_paths[i]
        BS_errors[i] = BS_delta_experiment_error(path, opt_price, 0.2, 0.06, 40, 1, boundary_prices)

    plt.hist(BS_errors, density=True, bins=int((max(BS_errors)-min(BS_errors))/0.14), edgecolor='white')
    plt.xlabel("Hedging-error")
    plt.ylabel("Density")
    plt.xlim(-3.6, 3.6)
    plt.ylim(0, 1.7)

    return plt.show()


def plot_BS_delta_errors_1(experiment_paths, boundary_prices, opt_price):

    BS_errors = np.zeros(len(experiment_paths))
    to_maturity = np.zeros(len(experiment_paths))

    for i in range(len(experiment_paths)):
        path = experiment_paths[i]
        BS_errors[i], to_maturity[i] = BS_delta_experiment_error(path, opt_price, 0.2, 0.06, 40,
                                                                 1, boundary_prices)

    BS_errors_T = np.where(to_maturity>0,BS_errors,0)
    BS_errors_T = BS_errors_T[BS_errors_T != 0]

    BS_errors_exercise = np.where(to_maturity==0,BS_errors,0)
    BS_errors_exercise = BS_errors_exercise[BS_errors_exercise != 0]

    plt.hist(BS_errors_T, density=True, bins=int((max(BS_errors_T)-min(BS_errors_T))/0.14), edgecolor='white', alpha=0.8)
    plt.hist(BS_errors_exercise, density=True, bins=int((max(BS_errors_exercise) - min(BS_errors_exercise)) / 0.14),
             edgecolor='white', alpha=0.8)
    plt.xlabel("Hedging-error")
    plt.ylabel("Density")
    plt.xlim(-3.6, 3.6)
    plt.ylim(0, 1.7)
    plt.legend(["To maturity", "Exercised before maturity"])

    return plt.show()


def plot_ISD_delta_fun_1(boundary_prices):
    prices = np.arange(20, 70, 0.5)
    bin_deltas = np.zeros(len(prices))

    ISD_deltas_1 = np.zeros(len(prices))
    ISD_deltas_2 = np.zeros(len(prices))
    ISD_deltas_3 = np.zeros(len(prices))

    for i in range(len(prices)):
        S, V = binomial_values(prices[i], 0.2, 0.06, 40, 1, 50)
        bin_deltas[i] = binomial_delta(S, V)

        ISD_paths_1 = ISD_paths_GBM_dt(0.06, 0.2, prices[i], 100000, 50, 0.02, 5)
        X1, Y1 = naive_vf_method_dt(ISD_paths_1, 0.06, 40, 0.02, 50, boundary_prices, 8)
        ISD_deltas_1[i] = naive_vf_delta(X1, Y1, prices[i], 8)

        ISD_paths_2 = ISD_paths_GBM_dt(0.06, 0.2, prices[i], 50000, 50, 0.02, 5)
        X2, Y2 = naive_vf_method_dt(ISD_paths_2, 0.06, 40, 0.02, 50, boundary_prices, 8)
        ISD_deltas_2[i] = naive_vf_delta(X2, Y2, prices[i], 8)

        ISD_paths_3 = ISD_paths_GBM_dt(0.06, 0.2, prices[i], 25000, 50, 0.02, 5)
        X3, Y3 = naive_vf_method_dt(ISD_paths_3, 0.06, 40, 0.02, 50, boundary_prices, 8)
        ISD_deltas_3[i] = naive_vf_delta(X3, Y3, prices[i], 8)

    plt.plot(prices, bin_deltas, linewidth=1.5)
    plt.scatter(prices, ISD_deltas_1, s=8, color='#ff7f0e')
    plt.scatter(prices, ISD_deltas_2, s=6, color='#2ca02c')
    plt.scatter(prices, ISD_deltas_3, s=6, color='#e377c2')

    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Delta")
    plt.legend(["Binomial", "Paths: 100000", "Paths: 50000", "Paths: 25000"])

    return plt.show()


def plot_ISD_delta_errors_1(experiment_paths, boundary_prices, opt_price):

    ISD_errors = np.zeros(len(experiment_paths))

    for i in range(500, 1000):
        path = experiment_paths[i]
        ISD_errors[i] = ISD_delta_experiment_error_1(path, opt_price, 0.2, 0.06, 40, 1,
                                                     boundary_prices, 5, 8, 8)

    plt.hist(ISD_errors, density=True, bins=int((max(ISD_errors)-min(ISD_errors))/0.14), edgecolor='white')
    plt.ylabel("Density")
    plt.xlabel("Hedging-error")
    plt.xlim(-3.6, 3.6)
    plt.ylim(0, 1.7)

    return plt.show()


def plot_ISD_delta_errors_2(experiment_paths, boundary_prices, opt_price):

    ISD_errors = np.zeros(len(experiment_paths))

    ISD_paths = ISD_paths_GBM(0.06, 0.2, 36, 250000, 50, 1, 5)
    coefficients = ISD_delta_functions_coef(ISD_paths, 36, 8, 0.06, 1, 40, boundary_prices, 8)

    for i in range(0, 1000):
        path = experiment_paths[i]
        ISD_errors[i] = ISD_delta_experiment_error_2(path, opt_price, coefficients, 0.06, 0.2, 36,
                                                     40, 1, boundary_prices)

    plt.hist(ISD_errors, density=True, bins=int((max(ISD_errors)-min(ISD_errors))/0.14), edgecolor='white')
    plt.ylabel("Density")
    plt.xlabel("Hedging-error")
    plt.xlim(-3.6, 3.6)
    plt.ylim(0, 1.7)

    return plt.show()


def plot_ISD_delta_fun_2(T):
    prices = np.arange(20, 70, 0.5)
    bin_deltas = np.zeros(len(prices))
    ISD_deltas = np.zeros(len(prices))

    ISD_paths = ISD_paths_GBM(0.06, 0.2, 36, 100000, 50, 1, 5)
    coefficients = ISD_delta_functions_coef(ISD_paths, 36, 8, 0.06, 1, 40, boundary_prices, 8)

    for i in range(len(prices)):
        S, V = binomial_values_dt(prices[i], 0.2, 0.06, 40, T/50, 50)
        bin_deltas[i] = binomial_delta(S, V)
        ISD_deltas[i] = ISD_delta_function(coefficients[:, int(50-T/0.02)], prices[i], 36)

    plt.plot(prices, bin_deltas)
    plt.plot(prices, ISD_deltas)
    plt.ylim(-1.1,0.1)
    plt.xlabel("Price of the underlying stock")
    plt.ylabel("Delta")

    return plt.show()


# boxplot
a = pd.DataFrame({ 'group' : np.repeat('BM',1000), 'value': BM_errors})
b = pd.DataFrame({ 'group' : np.repeat('LSM_1',500), 'value': LSM_errors_1})
c = pd.DataFrame({ 'group' : np.repeat('LSM_2',500), 'value': LSM_errors_2})
d = pd.DataFrame({ 'group' : np.repeat('BS',1000), 'value': BS_errors})
e = pd.DataFrame({ 'group' : np.repeat('ISD_1',1000), 'value': ISD_errors_1})
f = pd.DataFrame({ 'group' : np.repeat('ISD_2',1000), 'value': ISD_errors_2})
df = pd.concat((a,b,c,d,e,f))
sns.boxplot(x='group', y='value', data=df, palette="bwr")
plt.grid(color='whitesmoke')
plt.show()

