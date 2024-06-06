import numpy as np


def binomial_values_dt(S_0, sigma, r, strike, dt, total_steps):
    """
    Function which creates matrix with prices of the underlying stock.
    In this function we know the total number of steps and the length of each step.
    :param S_0: initial stock price
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: strike price
    :param dt: length of each time step
    :param total_steps: total number of steps
    :return: matrix S
    """
    # u and d in the binomial model
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    # the risk-neutral probabilities
    q_u = (np.exp(r*dt)-d) / (u-d)
    q_d = (u-np.exp(r*dt)) / (u-d)

    # binomial tree of the price of the underlying asset
    S = np.zeros((total_steps+1, total_steps+1))
    S[-1, 0] = S_0
    for i in range(1, total_steps+1):
        S[-1, i] = S[-1, i-1] * d
    for i in reversed(range(0, total_steps)):
        for j in range(1, total_steps+1):
            S[i, j] = S[i+1, j-1] * u

    # matrix containing the value of the american put option
    V = np.zeros((total_steps+1, total_steps+1))
    V[:, -1] = np.maximum(strike-S[:, -1], 0)
    for i in reversed(range(0, total_steps)):
        for j in range(total_steps-i, total_steps+1):
            V[j, i] = np.maximum(np.maximum(strike-S[j, i], 0), np.exp(-r*dt) * (q_u*V[j-1, i+1] + q_d*V[j, i+1]))

    return S, V


def binomial_values(S_0, sigma, r, strike, T, steps):
    """
    Function to find the lattice and the value of an American put option using the binomial model at each node.
    :param S_0: initial price of the underlying asset
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :param steps: number of steps pr. year
    :return: matrix V
    """
    # length of each time step
    dt = 1/steps
    # u and d in the binomial model
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    # the risk-neutral probabilities
    q_u = (np.exp(r*dt)-d) / (u-d)
    q_d = (u-np.exp(r*dt)) / (u-d)

    # binomial tree of the price of the underlying asset
    S = np.zeros((steps*T+1, steps*T+1))
    S[steps*T, 0] = S_0
    for i in range(1,steps*T+1):
        S[steps*T, i] = S[steps*T, i-1] * d
    for i in reversed(range(0, steps*T)):
        for j in range(1, steps*T+1):
            S[i, j] = S[i+1, j-1] * u

    # matrix containing the value of the american put option
    V = np.zeros((steps*T+1, steps*T+1))
    V[:, -1] = np.maximum(strike-S[:, -1], 0)
    for i in reversed(range(0, steps*T)):
        for j in range(steps*T-i, steps*T+1):
            V[j, i] = np.maximum(np.maximum(strike-S[j, i], 0), np.exp(-r*dt) * (q_u*V[j-1, i+1] + q_d*V[j, i+1]))

    return S, V


def binomial_price(V):
    """
    Function to determine the price
    :param S: matrix with prices of the underlying stock
    :param V: matrix with values of the option at each time step
    :return: price of the american put option
    """
    price = V[-1, 0]

    return price


def binomial_boundary(S_0, sigma, r, strike, T, steps):
    """
    Determine vector X (containing time steps) and Y (containing the stock prices corresponding to the boundary)
    With these one can plot the early exercise boundary
    :param S_0: initial price of the underlying asset
    :param sigma: volatility
    :param r: risk-free rate
    :param strike: the strike price
    :param T: maturity date
    :param steps: number of steps pr. year
    :return: X and Y
    """
    dt = 1 / steps
    S, V = binomial_values(S_0, sigma, r, strike, T, steps)

    # matrix showing the exercise boundary indicated by ones
    exercise_matrix = np.zeros((steps*T+1, steps*T+1))
    for i in reversed(range(0, steps*T+1)):
        for j in range(steps*T-i, steps*T+1):
            exercise_matrix[j, i] = np.where(V[j, i] == strike-S[j, i], 1, 0)
        for k in reversed(range(steps*T-i, steps*T+1)):
            if exercise_matrix[k-1, i] == 0:
                break
            else:
                exercise_matrix[k, i] = np.where(exercise_matrix[k-1, i] == 1, 0, 1)

    # vector from the first timestep the holder wants to exercise the option until maturity
    X = np.arange(dt, T+dt, dt)
    # matrix with prices of the underlying asset corresponding to the exercise boundary
    S_exercise_boundary = np.where(exercise_matrix == 1, S, 0)
    # vector containing the prices
    Y = np.zeros(T*steps)
    for i in range(T*steps):
        Y[i] = max(S_exercise_boundary[:, i+1])
    Y[Y == 0] = np.nan

    return X, Y


def binomial_delta(S, V):
    """
    Function to determine the delta
    :param S: matrix with prices of the underlying stock
    :param V: matrix with values of the option at each time step
    :return: delta of the american put option
    """
    delta = (V[-2, 1] - V[-1, 1]) / (S[-2, 1] - S[-1, 1])

    return delta
