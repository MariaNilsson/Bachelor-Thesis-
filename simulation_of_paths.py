import numpy as np


def monte_carlo_GBM(r, sigma, S_0, num_paths, num_steps, T):
    """
    Simulating multiple paths of an underlying asset with initial price S_0.
    :param r: risk-free rate
    :param sigma: volatility
    :param S_0: initial price of the underlying asset
    :param num_paths: number of paths
    :param num_steps: number of time steps pr. year
    :param T: number of years
    :return: a matrix with simulated paths of the price of an underlying asset.
    """
    # generating a matrix with zeroes to store the paths
    S = np.zeros((num_paths, T * num_steps + 1))
    # at time t=0 the price is S_0 for each path
    S[:, 0] = S_0
    # calculating the length of each time step
    dt = 1/num_steps
    # for each path
    for i in range(0, num_paths):
        # for each time step
        for j in range(1, T * num_steps + 1):
            Z = np.random.normal(0, 1)
            S[i, j] = S[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S


def monte_carlo_GBM_dt(r, sigma, S_0, num_paths, dt, total_steps):
    """
    Simulating multiple paths of an underlying asset with initial price S_0.
    :param r: risk-free rate
    :param sigma: volatility
    :param S_0: initial price of the underlying asset
    :param num_paths: number of paths
    :param total_steps: number of total steps
    :return: a matrix with simulated paths of the price of an underlying asset.
    """
    # generating a matrix with zeroes to store the paths
    S = np.zeros((num_paths, total_steps + 1))
    # at time t=0 the price is S_0 for each path
    S[:, 0] = S_0
    # for each path
    for i in range(0, num_paths):
        # for each time step
        for j in range(1, total_steps + 1):
            Z = np.random.normal(0, 1)
            S[i, j] = S[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return S


def ISD_paths_GBM(r, sigma, x_0, num_paths, num_steps, T, alpha):
    """
    Simulate stock price paths using ISD and GBM
    :param r: risk-free rate
    :param sigma: volatility
    :param x_0: initial stock price
    :param num_paths: number of paths
    :param num_steps: number of steps
    :param T: number of years
    :param alpha: ISD-parameter
    :return: matrix with simulated stock prices
    """
    N = num_paths
    X = np.zeros(N)
    U = np.random.uniform(0,1,N)
    for i in range(N):
        X[i] = x_0 + alpha*2*np.sin(np.arcsin(2*U[i]-1)/3)

    # generating a matrix with zeroes to store the paths
    S = np.zeros((num_paths, T * num_steps + 1))
    # at time t=0 the price is S_0 for each path
    S[:, 0] = X
    # calculating the length of each time step
    dt = 1/num_steps
    # for each path
    for i in range(0, num_paths):
        # for each time step
        for j in range(1, T * num_steps + 1):
            Z = np.random.normal(0, 1)
            S[i, j] = S[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return S


def ISD_paths_GBM_dt(r, sigma, x_0, num_paths, total_steps, dt, alpha):
    """
    Simulate stock price paths using ISD and GBM
    :param r: risk-free rate
    :param sigma: volatility
    :param x_0: initial stock price
    :param num_paths: number of paths
    :param total_steps: number of total steps
    :param dt: length of each time steps
    :param alpha: ISD-parameter
    :return: matrix with simulated stock prices
    """
    N = num_paths
    X = np.zeros(N)
    U = np.random.uniform(0,1,N)
    for i in range(N):
        X[i] = x_0 + alpha*2*np.sin(np.arcsin(2*U[i]-1)/3)

    # generating a matrix with zeroes to store the paths
    S = np.zeros((num_paths, total_steps + 1))
    # at time t=0 the price is S_0 for each path
    S[:, 0] = X
    # for each path
    for i in range(0, num_paths):
        # for each time step
        for j in range(1, total_steps + 1):
            Z = np.random.normal(0, 1)
            S[i, j] = S[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return S
