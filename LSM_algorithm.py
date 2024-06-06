import numpy as np
from sklearn.linear_model import LinearRegression
from Final.simulation_of_paths import *


def LSM_algorithm(sim_paths, r, strike, T):
    """
    Function to determine optimal exercise strategy and corresponding cash-flow for each path
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :return: an estimated price of the american option
    """
    # dividing the simulated paths by the strike price
    sim_paths = sim_paths / strike
    # number of paths
    num_paths = len(sim_paths)
    # number of time intervals
    num_steps = len(sim_paths[0]) - 1
    # length of each time step
    dt = T / num_steps
    # discount factors
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)
    # creating matrix with zeroes to store cashflows of the optimal strategy
    C = np.zeros((num_paths, num_steps))
    C[:, -1] = np.maximum(1 - sim_paths[:, -1], 0)
    # backwards iterations for each timestep
    for i in reversed(range(1, num_steps)):
        # the value of immediate exercise at time t = i
        exercise_value = np.maximum(1 - sim_paths[:, i], 0)
        # paths for which the option is ITM at time t = i
        ITM = exercise_value > 0
        # defining the vector X with the underlying asset price for the paths ITM
        X = sim_paths[:, i][ITM]
        # defining the vector Y with the corresponding discounted cash-flows
        Y = np.array(C[:, i:].dot(disc_vector[:(num_steps - i)]))
        Y = Y[ITM]
        # creating matrix with zeroes to store Laguerre values (we have three basis functions)
        model_matrix = np.zeros((len(X), 3))
        model_matrix[:, 0] = np.exp(-X / 2)
        model_matrix[:, 1] = np.exp(-X / 2) * (1 - X)
        model_matrix[:, 2] = np.exp(-X / 2) * (1 - 2 * X + (X ** 2) / 2)
        model = LinearRegression().fit(model_matrix, Y)
        continuation = np.zeros(num_paths)
        continuation[ITM] = model.predict(model_matrix)
        # optimal exercise strategy at t=i (True: exercise, False: continuation)
        strategy = ITM & (exercise_value > continuation)
        # update cashflow matrix C
        C[:, i - 1][strategy] = exercise_value[strategy]
        for j in range(i, num_steps):
            C[:, j][strategy] = 0

    C = C * strike

    return C


def LSM_algorithm_dt(sim_paths, r, strike, dt):
    """
    Function to determine optimal exercise strategy and corresponding cash-flow for each path.
    In this function dt is known.
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param dt: length of each time step
    :return: an estimated price of the american option
    """
    # dividing the simulated paths by the strike price
    sim_paths = sim_paths / strike
    # number of paths
    num_paths = len(sim_paths)
    # number of time intervals
    num_steps = len(sim_paths[0]) - 1
    # discount factors
    t = np.arange(dt, dt*num_steps+dt, dt)
    disc_vector = np.exp(-r * t)
    # creating matrix with zeroes to store cashflows of the optimal strategy
    C = np.zeros((num_paths, num_steps))
    C[:, -1] = np.maximum(1 - sim_paths[:, -1], 0)
    # backwards iterations for each timestep
    for i in reversed(range(1, num_steps)):
        # the value of immediate exercise at time t = i
        exercise_value = np.maximum(1 - sim_paths[:, i], 0)
        # paths for which the option is ITM at time t = i
        ITM = exercise_value > 0
        # defining the vector X with the underlying asset price for the paths ITM
        X = sim_paths[:, i][ITM]
        # defining the vector Y with the corresponding discounted cash-flows
        Y = np.array(C[:, i:].dot(disc_vector[:(num_steps - i)]))
        Y = Y[ITM]
        # creating matrix with zeroes to store Laguerre values (we have three basis functions)
        model_matrix = np.zeros((len(X), 3))
        model_matrix[:, 0] = np.exp(-X / 2)
        model_matrix[:, 1] = np.exp(-X / 2) * (1 - X)
        model_matrix[:, 2] = np.exp(-X / 2) * (1 - 2 * X + (X ** 2) / 2)
        model = LinearRegression().fit(model_matrix, Y)
        continuation = np.zeros(num_paths)
        continuation[ITM] = model.predict(model_matrix)
        # optimal exercise strategy at t=i (True: exercise, False: continuation)
        strategy = ITM & (exercise_value > continuation)
        # update cashflow matrix C
        C[:, i - 1][strategy] = exercise_value[strategy]
        for j in range(i, num_steps):
            C[:, j][strategy] = 0

    C = C * strike

    return C


def LSM_price(sim_paths, r, strike, T):
    """
    Function to determine the price of an american put option using the LSM-algorithm
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :return: price
    """
    # cash-flow matrix
    C = LSM_algorithm(sim_paths, r, strike, T)
    # number of time intervals
    num_steps = len(sim_paths[0]) - 1
    # length of each time step
    dt = T / num_steps
    # discount factors
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)
    c = np.array(C.dot(disc_vector))
    # calculating the price as the average of the discounted cashflows
    price = np.average(c)

    return price


def LSM_boundary(sim_paths, r, strike, T):
    """
    Determine vector time_steps (containing time steps) and boundary_prices (containing the stock prices corresponding
    to the boundary). With these vectors one can plot the early exercise boundary.
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: strike price
    :param T: maturity date
    :return: time_steps and boundary_prices
    """
    # cash-flow matrix
    C = LSM_algorithm(sim_paths, r, strike, T)
    # number of total steps
    num_steps = len(sim_paths[0]) - 1
    # length of each time step
    dt = T / num_steps

    exercise_boundary = np.where(C > 0, np.delete(sim_paths, 0, 1), 0)
    boundary_prices = np.zeros(num_steps)
    for i in range(num_steps):
        boundary_prices[i] = max(exercise_boundary[:, i])
    boundary_prices[boundary_prices == 0] = np.nan

    time_steps = np.arange(dt, T + dt, dt)

    return time_steps, boundary_prices


def LSM_boundary_price(sim_paths, boundary_prices, strike, dt, r):

    if len(sim_paths[0])-1 != len(boundary_prices):
        print("number of steps does not equal number of boundary prices")

    num_steps = len(sim_paths[0])-1

    # get matrix with cash-flows for each path following the LSM boundary
    C = np.zeros([len(sim_paths), num_steps])
    for i in range(0, len(sim_paths)):
        for j in range(0, num_steps):
            if sim_paths[i, j + 1] <= boundary_prices[j]:
                C[i, j] = strike - sim_paths[i, j + 1]
                break
            else:
                C[i, j] = 0

    # discount factors
    t = np.arange(dt, dt * num_steps + 0.00001, dt)
    disc_vector = np.exp(-r * t)
    # discount back to start time
    c = np.array(C.dot(disc_vector))
    # calculating the price as the average of the discounted cashflows
    price = np.average(c)

    return price


def LSM_delta(boundary_prices, S_0, epsilon, r, sigma, strike, T, num_paths, num_steps):
    """
    Function to determine delta, where the boundary is determined for both simulated paths.
    :param S_0: initial stock price
    :param epsilon: small distance
    :param r: risk-free rate
    :param sigma: volatility
    :param strike: strike price
    :param T: time to maturity
    :param num_paths: number of paths
    :param num_steps: number of steps per year
    :return:
    """
    # length of each time step
    dt = 1/num_steps

    # simulate paths
    sim_paths = monte_carlo_GBM(r, sigma, S_0, num_paths, num_steps, T)
    sim_paths_epsilon = monte_carlo_GBM(r, sigma, S_0 + epsilon, num_paths, num_steps, T)

    # price for sim_path
    price = LSM_boundary_price(sim_paths, boundary_prices, strike, dt, r)
    # price for sim_paths_epsilon
    price_epsilon = LSM_boundary_price(sim_paths_epsilon, boundary_prices, strike, dt, r)

    delta = (price_epsilon - price) / epsilon

    return delta


def LSM_delta_dt(boundary_prices, S_0, epsilon, r, sigma, strike, dt, num_paths, num_steps):
    """
    Function to determine delta, where the boundary is determined for both simulated paths.
    :param S_0: initial stock price
    :param epsilon: small distance
    :param r: risk-free rate
    :param sigma: volatility
    :param strike: strike price
    :param T: time to maturity
    :param num_paths: number of paths
    :param num_steps: number of steps per year
    :return:
    """
    # simulated paths with initial price path[i]
    sim_paths = monte_carlo_GBM_dt(r, sigma, S_0, num_paths, dt, num_steps)
    # simulated paths with initial price path[i] + epsilon
    sim_paths_epsilon = monte_carlo_GBM_dt(r, sigma, S_0 + epsilon, num_paths, dt, num_steps)

    # price for sim_path
    price = LSM_boundary_price(sim_paths, boundary_prices, strike, dt, r)
    # price for sim_paths_epsilon
    price_epsilon = LSM_boundary_price(sim_paths_epsilon, boundary_prices, strike, dt, r)

    # deltas
    delta = (price_epsilon - price) / epsilon

    return delta


def LSM_algorithm_poly(sim_paths, r, strike, T, deg):
    """
    Function to calculate the numbers in table 2 (here we use Laguerre as basis functions)
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param deg: degree of polynomial
    :return: an estimated price of the american option
    """
    # dividing the simulated paths by the strike price
    sim_paths = sim_paths / strike
    # number of paths
    num_paths = len(sim_paths)
    # number of time intervals
    num_steps = len(sim_paths[0]) - 1
    # length of each time step
    dt = T / num_steps
    # discount factors
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)
    # creating matrix with zeroes to store cashflows of the optimal strategy
    C = np.zeros((num_paths, num_steps))
    C[:, -1] = np.maximum(1 - sim_paths[:, -1], 0)
    # backwards iterations for each timestep
    for i in reversed(range(1, num_steps)):
        # the value of immediate exercise at time t = i
        exercise_value = np.maximum(1 - sim_paths[:, i], 0)
        # paths for which the option is ITM at time t = i
        ITM = exercise_value > 0
        # defining the vector X with the underlying asset price for the paths ITM
        X = sim_paths[:, i][ITM]
        # defining the vector Y with the corresponding discounted cash-flows
        Y = np.array(C[:, i:].dot(disc_vector[:(num_steps - i)]))
        Y = Y[ITM]
        # polyfit
        model = np.polyfit(X, Y, deg)
        p = np.poly1d(model)
        continuation = np.zeros(num_paths)
        continuation[ITM] = p(X)
        # optimal exercise strategy at t=i (True: exercise, False: continuation)
        strategy = ITM & (exercise_value > continuation)
        # update cashflow matrix C
        C[:, i - 1][strategy] = exercise_value[strategy]
        for j in range(i, num_steps):
            C[:, j][strategy] = 0

    C = C * strike

    return C


def LSM_algorithm_hermite(sim_paths, r, strike, T):
    """
    Function to determine optimal exercise strategy and corresponding cash-flow for each path
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :return: an estimated price of the american option
    """
    # dividing the simulated paths by the strike price
    sim_paths = sim_paths / strike
    # number of paths
    num_paths = len(sim_paths)
    # number of time intervals
    num_steps = len(sim_paths[0]) - 1
    # length of each time step
    dt = T / num_steps
    # discount factors
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)
    # creating matrix with zeroes to store cashflows of the optimal strategy
    C = np.zeros((num_paths, num_steps))
    C[:, -1] = np.maximum(1 - sim_paths[:, -1], 0)
    # backwards iterations for each timestep
    for i in reversed(range(1, num_steps)):
        # the value of immediate exercise at time t = i
        exercise_value = np.maximum(1 - sim_paths[:, i], 0)
        # paths for which the option is ITM at time t = i
        ITM = exercise_value > 0
        # defining the vector X with the underlying asset price for the paths ITM
        X = sim_paths[:, i][ITM]
        # defining the vector Y with the corresponding discounted cash-flows
        Y = np.array(C[:, i:].dot(disc_vector[:(num_steps - i)]))
        Y = Y[ITM]
        # creating matrix with zeroes to store Laguerre values (we have three basis functions)
        model_matrix = np.zeros((len(X), 3))
        model_matrix[:, 0] = 2*X
        model_matrix[:, 1] = 4*X**2-2
        model_matrix[:, 2] = 8*X**3-12*X
        model = LinearRegression().fit(model_matrix, Y)
        continuation = np.zeros(num_paths)
        continuation[ITM] = model.predict(model_matrix)
        # optimal exercise strategy at t=i (True: exercise, False: continuation)
        strategy = ITM & (exercise_value > continuation)
        # update cashflow matrix C
        C[:, i - 1][strategy] = exercise_value[strategy]
        for j in range(i, num_steps):
            C[:, j][strategy] = 0

    C = C * strike

    return C


def LSM_algorithm_legendre(sim_paths, r, strike, T):
    """
    Function to determine optimal exercise strategy and corresponding cash-flow for each path
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :return: an estimated price of the american option
    """
    # dividing the simulated paths by the strike price
    sim_paths = sim_paths / strike
    # number of paths
    num_paths = len(sim_paths)
    # number of time intervals
    num_steps = len(sim_paths[0]) - 1
    # length of each time step
    dt = T / num_steps
    # discount factors
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)
    # creating matrix with zeroes to store cashflows of the optimal strategy
    C = np.zeros((num_paths, num_steps))
    C[:, -1] = np.maximum(1 - sim_paths[:, -1], 0)
    # backwards iterations for each timestep
    for i in reversed(range(1, num_steps)):
        # the value of immediate exercise at time t = i
        exercise_value = np.maximum(1 - sim_paths[:, i], 0)
        # paths for which the option is ITM at time t = i
        ITM = exercise_value > 0
        # defining the vector X with the underlying asset price for the paths ITM
        X = sim_paths[:, i][ITM]
        # defining the vector Y with the corresponding discounted cash-flows
        Y = np.array(C[:, i:].dot(disc_vector[:(num_steps - i)]))
        Y = Y[ITM]
        # creating matrix with zeroes to store Laguerre values (we have three basis functions)
        model_matrix = np.zeros((len(X), 3))
        model_matrix[:, 0] = X
        model_matrix[:, 1] = 0.5*(3*X**2-1)
        model_matrix[:, 2] = 0.5*(5*X**3-3*X)
        model = LinearRegression().fit(model_matrix, Y)
        continuation = np.zeros(num_paths)
        continuation[ITM] = model.predict(model_matrix)
        # optimal exercise strategy at t=i (True: exercise, False: continuation)
        strategy = ITM & (exercise_value > continuation)
        # update cashflow matrix C
        C[:, i - 1][strategy] = exercise_value[strategy]
        for j in range(i, num_steps):
            C[:, j][strategy] = 0

    C = C * strike

    return C


def LSM_price_1(C, dt, r, T):
    """
    Function to determine the price of an american put option using the LSM-algorithm
    :param sim_paths: matrix with simulated paths
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :return: price
    """
    # discount factors
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-r * t)
    c = np.array(C.dot(disc_vector))
    # calculating the price as the average of the discounted cashflows
    price = np.average(c)

    return price
