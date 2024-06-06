import numpy as np
import statsmodels.api as sm


def naive_vf_method(ISD_paths, r, strike, T, boundary_prices, deg):
    """
    function to determine the initial price and the corresponding discounted cash-flows using the naive_vf method
    :param ISD_paths: matrix with ISD paths
    :param r: risk-free rate
    :param strike: strike price
    :param T: maturity date
    :param boundary_prices: vector of boundary prices
    :param deg: degree of polynomial used in regression at time t_1
    :return: initial prices and discounted cash-flows
    """
    # get matrix containing cash-flows for each ISD-path
    ISD_cashflows = np.zeros([len(ISD_paths), (len(ISD_paths[0])-2)])
    for i in range(0, len(ISD_paths)):
        for j in range(0, len(ISD_paths[0])-2):
            if ISD_paths[i, j+2] <= boundary_prices[j+1]:
                ISD_cashflows[i, j] = strike-ISD_paths[i, j+2]
                break
            else:
                ISD_cashflows[i, j] = 0

    # discount factors
    dt = T/(len(ISD_paths[0])-1)
    t = np.arange(dt, T, dt)
    disc_vector = np.exp(-r * t)

    # discounted payoffs (back to time t_1)
    Z = np.array(ISD_cashflows.dot(disc_vector))

    # defining the continuation function at time t=1
    X = ISD_paths[:, 1]

    model = np.polyfit(X, Z, deg)
    p = np.poly1d(model)
    continuation_value = p(X)

    exercise_value = np.maximum(strike - ISD_paths[:, 1], 0)

    value_function = np.zeros(len(ISD_paths))
    for i in range(0, len(ISD_paths)):
        if exercise_value[i] >= continuation_value[i]:
            value_function[i] = exercise_value[i]
        else:
            value_function[i] = continuation_value[i]

    # discount value function back to time t=0
    Y = value_function * np.exp(-r*dt)

    return ISD_paths[:, 0], Y


def naive_vf_method_dt(ISD_paths, r, strike, dt, total_steps, boundary_prices, deg):
    """
    function to determine the initial price and the corresponding discounted cash-flows using the naive_vf method
    :param ISD_paths: matrix with ISD paths
    :param r: risk-free rate
    :param strike: strike price
    :param dt: length of each time step
    :param total_steps: total steps
    :param boundary_prices: vector of boundary prices
    :param deg: degree of polynomial used in regression at time t_1
    :return: initial prices and discounted cash-flows
    """
    # get matrix containing cash-flows for each ISD-path
    ISD_cashflows = np.zeros([len(ISD_paths), (total_steps-1)])
    for i in range(0, len(ISD_paths)):
        for j in range(0, total_steps-1):
            if ISD_paths[i, j+2] <= boundary_prices[j+1]:
                ISD_cashflows[i, j] = strike-ISD_paths[i, j+2]
                break
            else:
                ISD_cashflows[i, j] = 0

    # discount factors
    t = np.arange(dt, dt*total_steps-0.001, dt)
    disc_vector = np.exp(-r * t)

    # discounted payoffs (back to time t_1)
    Z = np.array(ISD_cashflows.dot(disc_vector))

    # defining the continuation function at time t=1
    X = ISD_paths[:, 1]

    model = np.polyfit(X, Z, deg)
    p = np.poly1d(model)
    continuation_value = p(X)

    exercise_value = np.maximum(strike - ISD_paths[:, 1], 0)

    value_function = np.zeros(len(ISD_paths))
    for i in range(0, len(ISD_paths)):
        if exercise_value[i] >= continuation_value[i]:
            value_function[i] = exercise_value[i]
        else:
            value_function[i] = continuation_value[i]

    # discount value function back to time t=0
    Y = value_function * np.exp(-r*dt)

    # initial prices
    initial_prices = ISD_paths[:, 0]

    return initial_prices, Y


def naive_vf_coef(X, Y, x_0, M_0):
    """
    function to determine coefficients of the price function
    :param X: initial prices of the underlying stock
    :param Y: discounted cash-flows
    :param x_0: initial price
    :param M_0: degree of polynomial used in regression at time 0
    :return: coefficients of the price function
    """
    # price function
    model_matrix = np.zeros([len(Y), M_0 + 1])
    for i in range(0, M_0 + 1):
        model_matrix[:, i] = (X - x_0) ** i

    model = sm.OLS(Y, model_matrix)
    results = model.fit()
    params = results.params

    return params


def naive_vf_price_and_delta(X, Y, x_0, M_0):
    """
    function to determine the price and delta of the option
    :param X: initial prices of the underlying stock
    :param Y: discounted cash-flows
    :param x_0: initial price
    :param M_0: degree of polynomial used in regression at time 0
    :return: price and delta
    """
    # price function
    model_matrix = np.zeros([len(Y), M_0 + 1])
    for i in range(0, M_0 + 1):
        model_matrix[:, i] = (X - x_0) ** i

    model = sm.OLS(Y, model_matrix)
    results = model.fit()
    params = results.params

    price = params[0]
    delta = params[1]

    return price, delta


def naive_vf_delta(X, Y, x_0, M_0):
    """
    function to determine the delta of the option
    :param X: initial prices of the underlying stock
    :param Y: discounted cash-flows
    :param x_0: initial price
    :param M_0: degree of polynomial used in regression at time 0
    :return: delta
    """
    # price function
    model_matrix = np.zeros([len(Y), M_0 + 1])
    for i in range(0, M_0 + 1):
        model_matrix[:, i] = (X - x_0) ** i

    model = sm.OLS(Y, model_matrix)
    results = model.fit()
    params = results.params

    delta = params[1]

    return delta
