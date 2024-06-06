import numpy as np
from scipy.stats import norm


def black_scholes_delta(S_0, T, K, sigma, r):
    """
    Function to determine the delta of a European put option.
    :param S_0: initial price
    :param T: maturity time
    :param K: strike price
    :param sigma: volatility
    :param r: risk-free rate
    :return: delta
    """
    delta = norm.cdf(1/(sigma*np.sqrt(T))*(np.log(S_0/K)+(r+1/2*sigma**2)*T)) - 1

    return delta
