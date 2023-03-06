#usr/bin/env python3
"""
Experiments for the paper "Bivariate beta distribution: parameter inference and diagnostics".

Author: Lucas Moschen

This script is a support for the paper "Bivariate beta distribution: parameter inference and diagnostics" 
from Lucas Machado Moschen and Luiz Max Carvalho. 
It allows the user to replicate the results found in the paper.

This script requires that `numpy`, `scipy`, `lintegrate` and `tqdm` be installed within the Python 
environment you are running. 
"""

import numpy as np
from parameter_estimation import BivariateBeta
from tqdm import trange
from time import time
import os
import json
from __init__ import ROOT_DIR
from cmdstanpy import CmdStanModel

def starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed):
    """
    Prepares the experiment file for the well-specified case, that is, the data comes from
    the Bivariate Beta distribution.
    """
    filename = '../../experiments/exp_' + '_'.join(str(e) for e in true_alpha) 
    filename += '_' + str(sample_size) + '_' + str(monte_carlo_size)
    filename += '_' + str(bootstrap_size) + '_' + str(seed) + '.json'
    filename = os.path.join(ROOT_DIR, filename)

    if not os.path.exists(filename):
        with open(filename, 'w') as outfile:
            data = {'n_experiments': 0, 'bias': 0, 'mse': 0, 'mape': 0, 'comp': 0, 'coverage': 0}
            json.dump(data, outfile)

    return filename

def starting_experiment_2(mu, sigma, sample_size, monte_carlo_size, seed):
    """
    Prepares the experiment file.
    """
    filename = '../../experiments/exp_logit_' + '_'.join(str(e) for e in mu) + '_' + '_'.join(str(e) for e in sigma.flatten())
    filename += '_' + str(sample_size) + '_' + str(monte_carlo_size)
    filename += '_' + str(seed) + '.json'
    filename = os.path.join(ROOT_DIR, filename)

    if not os.path.exists(filename):
        with open(filename, 'w') as outfile:
            data = {'n_experiments': 0, 'bias': 0, 'mse': 0, 'mape': 0, 'comp': 0, 'coverage': 0}
            json.dump(data, outfile)

    return filename

def saving_document_1(filename, bias, mse, mape, comp, coverage):
    """
    Saves the information for each experiment in the well-specified case, that is, the data comes from
    the Bivariate Beta distribution.
    """
    with open(filename, 'r') as outfile:
        data = json.load(outfile)

    N = data['n_experiments']
    bias = (np.array(data['bias']) * N + bias)/(N + 1)
    mse = (np.array(data['mse']) * N + mse)/(N + 1)
    mape = (np.array(data['mape']) * N + mape)/(N + 1)
    comp = (np.array(data['comp']) * N + comp)/(N + 1)
    coverage = (np.array(data['coverage']) * N + coverage)/(N + 1)

    data['n_experiments'] += 1
    data['bias'] = bias.tolist()
    data['mse'] = mse.tolist()
    data['mape'] = mape.tolist()
    data['comp'] = comp.tolist()
    data['coverage'] = coverage.tolist()

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def saving_document_2(filename, bias, mse, mape):
    """
    Saves the information for each experiment
    """
    with open(filename, 'r') as outfile:
        data = json.load(outfile)

    N = data['n_experiments']
    bias = (np.array(data['bias']) * N + bias)/(N + 1)
    mse = (np.array(data['mse']) * N + mse)/(N + 1)
    mape = (np.array(data['mape']) * N + mape)/(N + 1)

    data['n_experiments'] += 1
    data['bias'] = bias.tolist()
    data['mse'] = mse.tolist()
    data['mape'] = mape.tolist()

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def experiment_bivbeta(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed, coverage=True):
    """
    It does the experiments from Section "Recovering parameters from bivariate beta".
    """
    coverage_new = np.zeros((5,4))

    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()

    filename = starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed)

    # Stan setting
    data = {'n': sample_size, 'a': np.ones(4), 'b': np.ones(4)}
    stanfile = os.path.join(ROOT_DIR, '..', 'stan', 'bivariate-beta-model-v3.stan')
    model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': True})

    for _ in trange(monte_carlo_size):
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

        if coverage:

            methods = [distribution.method_moments_estimator_1, distribution.method_moments_estimator_2, 
                       distribution.method_moments_estimator_3, distribution.method_moments_estimator_4]
            alpha0_parameters = [None, None, (1,1), (1,1,1,1)]

            for ind in range(4):
                samples = distribution.bootstrap_method_parametric(x=X, y=Y, 
                                                        B=bootstrap_size,
                                                        method=methods[ind],
                                                        processes=4,
                                                        seed=rng.integers(2**32-1),
                                                        alpha0=alpha0_parameters[ind],
                                                        x0=None)
                ci = distribution.confidence_interval(level=0.95, samples=samples)
                coverage_new[ind, :] = (ci[0,:] < true_alpha)*(ci[1,:] > true_alpha)
        
        t0 = time()
        data['xy'] = np.column_stack([X,Y])
        model_fit = model.sample(data=data, iter_warmup=2000, iter_sampling=2000, chains=4, adapt_delta=0.9, 
                                 show_progress=False, show_console=False)
        summary = model_fit.summary(percentiles=(2.5, 50, 97.5))
        alpha_hat5 = summary['Mean'].iloc[1:5].values
        alpha_hat6 = summary['50%'].iloc[1:5].values
        time5 = time() - t0
        
        alpha = np.array([alpha_hat1, alpha_hat2, alpha_hat3, alpha_hat4, alpha_hat5, alpha_hat6])
        lb = summary['2.5%'].iloc[1:5].values
        ub = summary['97.5%'].iloc[1:5].values
        coverage_new[4,:] = (lb < true_alpha)*(ub > true_alpha)

        # Updating the estimates iteratively
        bias_new = alpha - true_alpha
        mse_new = bias_new * bias_new
        mape_new = abs(bias_new)/true_alpha
        comp_new = np.array([time1, time2, time3, time4, time5])

        saving_document_1(filename, bias_new, mse_new, mape_new, comp_new, coverage_new)

def moments_logit_normal(mu, sigma):

    Z = np.random.multivariate_normal(mu, sigma, size=1000000)
    X = 1/(1 + np.exp(-Z))
    return np.array([X[:,0].mean(), X[:,1].mean(), 
                     X[:,0].var(ddof=1), X[:,1].var(ddof=1), 
                     np.corrcoef(X[:,0], X[:,1])[0,1]])

def experiment_logitnormal(mu, sigma, sample_size, monte_carlo_size, seed):
    """
    It does the experiments from Section "Recovering parameters from other bivariate distribution".
    """
    true_moments = moments_logit_normal(mu, sigma)

    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()

    filename = starting_experiment_2(mu, sigma, sample_size, monte_carlo_size, seed)

    # Stan setting
    data = {'n': sample_size, 'a': np.ones(4), 'b': np.ones(4)}
    stanfile = os.path.join(ROOT_DIR, '..', 'stan', 'bivariate-beta-model-v3.stan')
    model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': True})

    for exp in trange(monte_carlo_size):
        Z = rng.multivariate_normal(mu, sigma, size=sample_size)
        X = 1/(1 + np.exp(-Z[:, 0]))
        Y = 1/(1 + np.exp(-Z[:, 1]))
    
        alpha_hat1 = distribution.method_moments_estimator_1(X, Y)
        alpha_hat2 = distribution.method_moments_estimator_2(X, Y)
        alpha_hat3 = distribution.method_moments_estimator_3(X, Y, alpha0=(1, 1))
        alpha_hat4 = distribution.method_moments_estimator_4(X, Y, alpha0=(1, 1, 1, 1))
        
        data['xy'] = np.column_stack([X,Y])
        model_fit = model.sample(data=data, iter_warmup=2000, iter_sampling=2000, chains=4, adapt_delta=0.9, 
                                 show_progress=False, show_console=False)
        summary = model_fit.summary(percentiles=(2.5, 50, 97.5))
        alpha_hat5 = summary['Mean'].iloc[1:5].values
        alpha_hat6 = summary['50%'].iloc[1:5].values

        est_moments1 = BivariateBeta(alpha=alpha_hat1).moments()
        est_moments2 = BivariateBeta(alpha=alpha_hat2).moments()
        est_moments3 = BivariateBeta(alpha=alpha_hat3).moments()
        est_moments4 = BivariateBeta(alpha=alpha_hat4).moments()
        est_moments5 = BivariateBeta(alpha=alpha_hat5).moments()
        est_moments6 = BivariateBeta(alpha=alpha_hat6).moments()
        est_moments = np.array([est_moments1, est_moments2, est_moments3, est_moments4, est_moments5, est_moments6])

        # Updating the estimates iteratively
        bias_new = est_moments - true_moments
        mse_new = bias_new * bias_new
        mape_new = abs(bias_new)/abs(true_moments)

        saving_document_2(filename, bias_new, mse_new, mape_new)

def simulated_based_calibration(a, b, c, n, L=63, N=1000, seed=831290):

    a = a*np.ones(4)
    b = b*np.ones(4)
    c = 0.0

    data = {'n': n, 'a': a, 'b': b, 'c': c}

    rho_values = []
    rng = np.random.RandomState(seed)

    stanfile = os.path.join('..', '..', 'scripts', 'stan', 'bivariate-beta-model-v3.stan')
    model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': True})

    for _ in trange(2*N):
        # Data 
        true_alpha = rng.gamma(shape=a, scale=1/b, size=4) + c
        U = rng.dirichlet(true_alpha, size=n)
        if U[U<np.finfo(np.float64).eps].shape[0] > 0:
            continue
        X = U[:,0] + U[:,1]
        Y = U[:,0] + U[:,2]
        XY = np.column_stack([X,Y])
        data['xy'] = XY

        model_fit = model.sample(data=data, iter_warmup=1000, iter_sampling=1000, chains=1, adapt_delta=0.9,
                                 show_progress=False, show_console=False)
        alpha_estimates = (model_fit.stan_variables()['alpha'])[rng.choice(range(1000), size=L, replace=False)]
        rho = np.sum(alpha_estimates > true_alpha, axis=0)
        rho_values.append({'rho': rho.tolist(), 
                           'diagnose': model_fit.diagnose(),
                           'true_alpha': true_alpha.tolist(),
                           'XY': XY.tolist(),
                           'U': U.tolist()})

    folder = os.path.join(ROOT_DIR, '..', '..', 'experiments', 'sbc')
    if not os.path.exists(folder):
        os.mkdir(folder)
    name = "sbc_{}_{}_{}_{}_{}_{}_{}.json".format(a[0], b[0], c, n, L, N, seed)
    with open(os.path.join(folder, name), "w") as final:
        json.dump(rho_values, final)

if __name__ == '__main__':

    monte_carlo_size = 1000
    bootstrap_size = 500
    seed = 378291

    #true_alpha = np.array([1,1,1,1])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed)
    #experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed)

    #true_alpha = np.array([2,7,3,1])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed)
    #experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed)

    #true_alpha = np.array([0.7, 0.9, 2.0, 1.5])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed)
    #experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed)

    n = 50
    mu = np.array([0,0])
    sigma = np.array([[1.0, 0.1], [0.1, 1.0]])
    experiment_logitnormal(mu, sigma, n, monte_carlo_size, seed)

    mu = np.array([-1.0, -1.0])
    sigma = np.array([[2.25, -1.2], [-1.2, 1]])
    experiment_logitnormal(mu, sigma, n, monte_carlo_size, seed)

    # a = 1
    # b = 1
    # c = 0
    # n = 50
    # L = 63
    # N = 1000
    # seed = 831290
    # simulated_based_calibration(a, b, c, n, L, N, seed)




