#usr/bin/env python3
"""
Experiments for the paper "On a bivariate beta distribution".

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It allows the user to replicate 
the results found in the paper.

This script requires that `numpy`, `scipy`, `lintegrate` and `tqdm` be installed within the Python 
environment you are running. 
"""

import numpy as np
from parameter_estimation import BivariateBeta
from tqdm import tqdm, trange
from time import time
import os
import json
from __init__ import ROOT_DIR

def starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed):
    """
    Prepares the experiment file.
    """
    filename = '../experiments/exp_' + '_'.join(str(e) for e in true_alpha) 
    filename += '_' + str(sample_size) + '_' + str(monte_carlo_size)
    filename += '_' + str(bootstrap_size) + '_' + str(seed) + '.json'
    filename = os.path.join(ROOT_DIR, filename)

    if not os.path.exists(filename):
        with open(filename, 'w') as outfile:
            data = {'n_experiments': 0, 'bias': 0, 'mse': 0, 'comp': 0, 'coverage': 0}
            json.dump(data, outfile)

    return filename

def saving_document(filename, bias, mse, comp, coverage):
    """
    Saves the information for each experiment
    """
    with open(filename, 'r') as outfile:
        data = json.load(outfile)

    N = data['n_experiments']
    bias = (np.array(data['bias']) * N + bias)/(N + 1)
    mse = (np.array(data['mse']) * N + mse)/(N + 1)
    comp = (np.array(data['comp']) * N + comp)/(N + 1)
    coverage = (np.array(data['coverage']) * N + coverage)/(N + 1)

    data['n_experiments'] += 1
    data['bias'] = bias.tolist()
    data['mse'] = mse.tolist()
    data['comp'] = comp.tolist()
    data['coverage'] = coverage.tolist()

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def experiment_bivbeta(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed):
    """
    It does the experiments from Section "Recovering parameters from bivariate beta".
    """
    coverage_new = np.zeros((5,4))

    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()

    filename = starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed)

    for exp in trange(monte_carlo_size):
        U = rng.dirichlet(true_alpha, size=sample_size)
        X = U[:, 0] + U[:, 1]
        Y = U[:, 0] + U[:, 2]

        t0 = time()
        alpha_hat1 = distribution.method_moments_estimator_1(X, Y)
        time1 = time() - t0
        t0 = time()
        alpha_hat2 = distribution.method_moments_estimator_2(X, Y)
        time2 = time() - t0
        t0 = time()
        alpha_hat3 = distribution.method_moments_estimator_3(X, Y, alpha0=(1, 1))
        time3 = time() - t0
        t0 = time()
        alpha_hat4 = distribution.method_moments_estimator_4(X, Y, alpha0=(1, 1, 1, 1))
        time4 = time() - t0
        t0 = time()
        alpha_hat5 = distribution.modified_maximum_likelihood_estimator(X, Y, x0=(2, 2, 4))
        time5 = time() - t0
        alpha = np.array([alpha_hat1, alpha_hat2, alpha_hat3, alpha_hat4, alpha_hat5])

        # Updating the estimates iteratively
        bias_new = true_alpha - alpha
        mse_new = bias_new * bias_new
        comp_new = np.array([time1, time2, time3, time4, time5])

        nb = 0
        if exp < nb:

            methods = [distribution.method_moments_estimator_1, distribution.method_moments_estimator_2, 
                       distribution.method_moments_estimator_3, distribution.method_moments_estimator_4, 
                       distribution.modified_maximum_likelihood_estimator]
            alpha0_parameters = [None, None, (1,1), (1,1,1,1), None]
            x0_parameters = [None, None, None, None, (2,2,4)]

            for ind in range(5):
                samples = distribution.bootstrap_method(x=X, y=Y, 
                                                        B=bootstrap_size,
                                                        method=methods[ind], 
                                                        seed=rng.integers(9999999999),
                                                        alpha0=alpha0_parameters[ind],
                                                        x0=x0_parameters[ind])
                ci = distribution.confidence_interval(level=0.95, samples=samples)
                coverage_new[ind, :] = (ci[0,:] < true_alpha)*(ci[1,:] > true_alpha)
        
        saving_document(filename, bias_new, mse_new, comp_new, coverage_new)

def moments_logit_normal(mu, sigma):

    Z = np.random.normal(mu, sigma, size=1000000)
    X = 1/(1 + np.exp(-Z))
    return np.array([X[:,0].mean(), X[:,1].mean(), X[:,0].var(), X[:,1].var(), np.corrcoef(X[:,0], X[:,1])[0,1]])

def experiment_logitnormal(mu, sigma, sample_size, monte_carlo_simulations, bootstrap_sample_size, seed):

    true_moments = moments_logit_normal(mu, sigma)
    bias = np.zeros(4)
    mse = np.zeros(4)

    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()

    for exp in tqdm(range(monte_carlo_simulations)):
        Z = rng.multivariate_normal(mu, sigma, size=sample_size)
        X = 1/(1 + np.exp(-Z[:,0]))
        Y = 1/(1 + np.exp(-Z[:,1]))
    
        alpha_hat1 = distribution.method_moments_estimator_1(X, Y)
        alpha_hat2 = distribution.method_moments_estimator_2(X, Y)
        alpha_hat3 = distribution.method_moments_estimator_3(X, Y, alpha0=(1, 1))
        alpha_hat4 = distribution.method_moments_estimator_4(X, Y, alpha0=(1, 1, 1, 1))
        alpha =  np.array([alpha_hat1, alpha_hat2, alpha_hat3, alpha_hat4])

        estimated_bivbeta = BivariateBeta(alpha=alpha)
        estimated_moments = estimated_bivbeta.moments()

        bias_new = true_moments - estimated_moments
        mse_new = bias_new * bias_new
        bias = (bias * exp + bias_new)/(exp+1)
        mse = (mse * exp + mse_new)/(mse+1)

    return bias, mse

if __name__ == '__main__':

    true_alpha = np.array([1,1,1,1])
    sample_size = 50
    monte_carlo_size = 10000
    bootstrap_size = 500
    seed = 8392

    experiment_bivbeta(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed)