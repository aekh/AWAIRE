# This file is a modified version of a part of SHANGRLA.
# Copyright (C) 2019-2024 Philip B. Stark, Vanessa Teague, Michelle Blom,
#   Peter Stuckey, Ian Waudby-Smith, Jacob Spertus, Amanda Glazer,
#   Damjan Vukcevic, David Wu, Alexander Ek, Floyd Everest.
#
# See https://github.com/pbstark/SHANGRLA for more information.


import numpy as np
import scipy as sp
import math
import sys

def shrink_trunc(x: np.array, N: int, mu: float = 1 / 2, nu: float = 1 - np.finfo(float).eps, u: float = 1, \
                 c: float = 1 / 2, d: float = 100, f: float = 0, minsd: float = 10 ** -6, j0: int = 1, S0 : int = 0) -> np.array:
    '''
    apply the shrinkage and truncation estimator to an array
    sample mean is shrunk towards nu, with relative weight d compared to a single observation,
    then that combination is shrunk towards u, with relative weight f/(stdev(x)).
    The result is truncated above at u-u*eps and below at mu_j+e_j(c,j)
    The standard deviation is calculated using Welford's method.
    S_1 = 0
    S_j = \sum_{i=1}^{j-1} x_i, j > 1
    m_j = (N*mu-S_j)/(N-j+1) if np.isfinite(N) else mu
    e_j = c/sqrt(d+j-1)
    sd_1 = sd_2 = 1
    sd_j = sqrt[(\sum_{i=1}^{j-1} (x_i-S_j/(j-1))^2)/(j-2)] \wedge minsd, j>2
    eta_j =  ( [(d*nu + S_j)/(d+j-1) + f*u/sd_j]/(1+f/sd_j) \vee (m_j+e_j) ) \wedge u*(1-eps)
    eta_j =  ( [(d*nu + S_j)/(d+j-1) ] \vee (m_j+e_j) ) \wedge u*(1-eps)   if f=0
    Parameters
    ----------
    x : np.array
        input data
    mu : float in (0, 1)
        hypothesized population mean
    eta : float in (t, 1)
        initial alternative hypothethesized value for the population mean
    c : positive float
        scale factor for allowing the estimated mean to approach t from above
    d : positive float
        relative weight of nu compared to an observation, in updating the alternative for each term
    f : positive float
        relative weight of the upper bound u (normalized by the sample standard deviation)
    minsd : positive float
        lower threshold for the standard deviation of the sample, to avoid divide-by-zero errors and
        to limit the weight of u
    S0, j0 : for restarting mid-sequence
    '''
    S = np.insert(np.cumsum(x), 0, 0)[0:-1]+S0  # 0, x_1, x_1+x_2, ...,
    j = np.arange(j0, len(x) + j0)  # 1, 2, 3, ..., len(x)
    m = (N * mu - S) / (N - j + 1) if np.isfinite(N) else mu  # mean of population after (j-1)st draw, if null is true
    mj = [x[0]]  # Welford's algorithm for running mean and running SD
    sdj = [0]
    for i, xj in enumerate(x[1:]):
        mj.append(mj[-1] + (xj - mj[-1]) / (i + 1))
        sdj.append(sdj[-1] + (xj - mj[-2]) * (xj - mj[-1]))
    sdj = np.sqrt(sdj / j)
    sdj = np.insert(np.maximum(sdj, minsd), 0, 1)[0:-1]  # threshold the sd, set first sd to 1
    weighted = ((d * nu + S) / (d + j - 1) + f * u / sdj) / (1 + f / sdj)

    return np.minimum(u * (1 - np.finfo(float).eps), np.maximum(weighted, m + c / np.sqrt(d + j - 1))), m


def alpha_mart_old(x: np.array, N: int, mu: float=1/2, eta: float=1-np.finfo(float).eps, f: float=0, u: float=1, \
               estim: callable=shrink_trunc) -> np.array :
    '''
    Finds the ALPHA martingale for the hypothesis that the population
    mean is less than or equal to t using a martingale method,
    for a population of size N, based on a series of draws x.
    The draws must be in random order, or the sequence is not a martingale under the null
    If N is finite, assumes the sample is drawn without replacement
    If N is infinite, assumes the sample is with replacement
    Parameters
    ----------
    x : list corresponding to the data
    N : int
        population size for sampling without replacement, or np.infinity for sampling with replacement
    mu : float in (0,1)
        hypothesized fraction of ones in the population
    eta : float in (t,1)
        alternative hypothesized population mean
    estim : callable
        estim(x, N, mu, eta, u) -> np.array of length len(x), the sequence of values of eta_j for ALPHA
    Returns
    -------
    terms : array
        sequence of terms that would be a nonnegative supermartingale under the null
    '''
    S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
    j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
    m = (N*mu-S)/(N-j+1) if np.isfinite(N) else mu   # mean of population after (j-1)st draw, if null is true
    etaj = estim(x=x, N=N, mu=mu, nu=eta, f=f, u=u)
    with np.errstate(divide='ignore',invalid='ignore'):
        terms = np.cumprod((x*etaj/m + (u-x)*(u-etaj)/(u-m))/u)
    terms[m<0] = np.inf
    terms[m>u] = 0
    return terms, m


def alpha_mart(x: np.array, m : np.array, etaj : np.array, u: float=1, ) -> np.array :
    '''
    Modular ALPHA
    Parameters
    ----------
    x : list corresponding to the data
    m : list of mu's, same length as x
    etaj : list of eta's, same length as x
    Returns
    -------
    terms : array
        sequence of terms that would be a nonnegative supermartingale under the null
    '''
    with np.errstate(divide='ignore',invalid='ignore'):
        terms = np.cumprod((x*etaj/m + (u-x)*(u-etaj)/(u-m))/u)
    terms[m<0] = np.inf
    terms[m>u] = 0
    terms[m==u] = 1  # guard against division by 0
    terms[m==0] = 1  # guard against division by 0
    return terms, m
