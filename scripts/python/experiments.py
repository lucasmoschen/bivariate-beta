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

def starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-v3.stan', prior_a=None, prior_kappa=None, prior_lambda=None, prior_mu=None, prior_sd=None):
    """
    Prepares the experiment file for the well-specified case, that is, the data comes from
    the Bivariate Beta distribution.
    """
    filename = '../../experiments/exp_' + '_'.join(str(e) for e in true_alpha) 
    filename += '_' + str(sample_size) + '_' + str(monte_carlo_size)
    filename += '_' + str(bootstrap_size) + '_' + str(seed)
    
    if stan_model != 'bivariate-beta-model-v3.stan':
        filename += '_' + stan_model.split('.')[0]
        if prior_a is not None:
            filename += '_a' + ''.join(str(e) for e in prior_a)
        if prior_kappa is not None:
            filename += '_k' + str(prior_kappa)
        if prior_lambda is not None:
            filename += '_l' + str(round(prior_lambda, 3))
        if prior_mu is not None:
            filename += '_mu' + str(prior_mu)
        if prior_sd is not None:
            filename += '_sd' + str(prior_sd)
            
    filename += '.json'
    filename = os.path.join(ROOT_DIR, filename)

    if not os.path.exists(filename):
        with open(filename, 'w') as outfile:
            data = {'n_experiments': 0, 
                    'bias': 0, 'mse': 0, 'mape': 0, 
                    'bias_moments': np.zeros((6, 5)).tolist(),
                    'mse_moments': np.zeros((6, 5)).tolist(),
                    'mape_moments': np.zeros((6, 5)).tolist(),
                    'comp': np.zeros(5).tolist(),
                    'coverage': np.zeros((5, 4)).tolist()
                   }
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

def saving_document_1(filename, bias, mse, mape, comp, coverage, bias_moments, mse_moments, mape_moments):
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
    
    bias_m = (np.array(data.get('bias_moments', 0)) * N + bias_moments)/(N + 1)
    mse_m = (np.array(data.get('mse_moments', 0)) * N + mse_moments)/(N + 1)
    mape_m = (np.array(data.get('mape_moments', 0)) * N + mape_moments)/(N + 1)
    
    comp = (np.array(data['comp']) * N + comp)/(N + 1)
    coverage = (np.array(data['coverage']) * N + coverage)/(N + 1)

    data['n_experiments'] += 1
    data['bias'] = bias.tolist()
    data['mse'] = mse.tolist()
    data['mape'] = mape.tolist()
    data['bias_moments'] = bias_m.tolist()
    data['mse_moments'] = mse_m.tolist()
    data['mape_moments'] = mape_m.tolist()
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

def experiment_bivbeta(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed, coverage=True, stan_model='bivariate-beta-model-v3.stan', prior_a=None, prior_kappa=None, prior_lambda=None, prior_mu=None, prior_sd=None):
    """
    It does the experiments from Section "Recovering parameters from bivariate beta".
    """
    coverage_new = np.zeros((5,4))

    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()

    filename = starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed, stan_model, prior_a, prior_kappa, prior_lambda, prior_mu, prior_sd)

    # Stan setting
    if prior_a is None:
        prior_a = np.ones(4)
        
    data = {'n': sample_size, 'a': prior_a, 'b': np.ones(4)}
    if prior_kappa is not None:
        data['kappa'] = prior_kappa
    if prior_lambda is not None:
        data['lambda'] = prior_lambda
    if prior_mu is not None:
        data['s_mu'] = prior_mu
    if prior_sd is not None:
        data['s_sd'] = prior_sd

    stanfile = os.path.join(ROOT_DIR, '..', 'stan', stan_model)
    model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': True})

    runs_left = monte_carlo_size - json.load(open(filename, 'r')).get('n_experiments', 0)
    for _ in trange(runs_left):
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
        alpha_hat4 = distribution.method_moments_estimator_4(X, Y)
        time4 = time() - t0

        if coverage:

            methods = [distribution.method_moments_estimator_1, distribution.method_moments_estimator_2, 
                       distribution.method_moments_estimator_3, distribution.method_moments_estimator_4]
            alpha0_parameters = [None, None, (1,1), None]

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
        alpha_keys = [f'alpha[{i}]' for i in range(1, 5)]
        alpha_hat5 = summary.loc[alpha_keys, 'Mean'].values
        alpha_hat6 = summary.loc[alpha_keys, '50%'].values
        time5 = time() - t0
        
        alpha = np.array([alpha_hat1, alpha_hat2, alpha_hat3, alpha_hat4, alpha_hat5, alpha_hat6])
        lb = summary.loc[alpha_keys, '2.5%'].values
        ub = summary.loc[alpha_keys, '97.5%'].values
        coverage_new[4,:] = (lb < true_alpha)*(ub > true_alpha)

        # Updating the estimates iteratively
        bias_new = alpha - true_alpha
        mse_new = bias_new * bias_new
        mape_new = abs(bias_new)/true_alpha
        
        # Moment-based metrics
        true_moments = distribution.moments(true_alpha)
        est_moments = [BivariateBeta(alpha=a).moments() for a in alpha[:4]]
        alpha_samples = model_fit.stan_variables()['alpha']
        
        # Vectorized calculation of moments for all posterior samples
        s = alpha_samples.sum(axis=1)
        m1 = (alpha_samples[:,0] + alpha_samples[:,1]) / s
        m2 = (alpha_samples[:,0] + alpha_samples[:,2]) / s
        v1 = m1 * (alpha_samples[:,2] + alpha_samples[:,3]) / (s * (s+1))
        v2 = m2 * (alpha_samples[:,1] + alpha_samples[:,3]) / (s * (s+1))
        den = np.exp(-0.5 * (np.log(alpha_samples[:,0] + alpha_samples[:,1]) + 
                             np.log(alpha_samples[:,2] + alpha_samples[:,3]) + 
                             np.log(alpha_samples[:,0] + alpha_samples[:,2]) + 
                             np.log(alpha_samples[:,1] + alpha_samples[:,3])))
        cor = (alpha_samples[:,0] * alpha_samples[:,3] - alpha_samples[:,1] * alpha_samples[:,2]) * den
        
        moment_samples = np.column_stack([m1, m2, v1, v2, cor])
        
        est_moments.append(np.mean(moment_samples, axis=0))
        est_moments.append(np.median(moment_samples, axis=0))
        
        est_moments = np.array(est_moments)
        
        bias_m = est_moments - true_moments
        mse_m = bias_m * bias_m
        mape_m = abs(bias_m) / np.maximum(abs(true_moments), 1e-10)

        comp_new = np.array([time1, time2, time3, time4, time5])

        saving_document_1(filename, bias_new, mse_new, mape_new, comp_new, coverage_new, bias_m, mse_m, mape_m)

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
        alpha_hat4 = distribution.method_moments_estimator_4(X, Y)
        
        data['xy'] = np.column_stack([X,Y])
        model_fit = model.sample(data=data, iter_warmup=2000, iter_sampling=2000, chains=4, adapt_delta=0.9, 
                                 show_progress=False, show_console=False)
        summary = model_fit.summary(percentiles=(2.5, 50, 97.5))
        alpha_keys = [f'alpha[{i}]' for i in range(1, 5)]
        alpha_hat5 = summary.loc[alpha_keys, 'Mean'].values
        alpha_hat6 = summary.loc[alpha_keys, '50%'].values

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

def simulated_based_calibration_lognormal(a, s_mu, s_sd, n, L=63, N=1000, seed=831290):

    a = a*np.ones(4)

    data = {'n': n, 'a': a, 's_mu': s_mu, 's_sd': s_sd}

    rho_values = []
    rng = np.random.RandomState(seed)

    stanfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stan', 'bivariate-beta-model-lognormal.stan'))
    model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': True})

    for _ in trange(2*N):
        # Data 
        s = rng.lognormal(mean=np.log(s_mu), sigma=s_sd)
        theta = rng.dirichlet(a)
        true_alpha = s * theta
        
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

    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'sbc'))
    if not os.path.exists(folder):
        os.mkdir(folder)
    name = f"sbc_lognormal_a{a[0]}_mu{s_mu}_sd{s_sd}_{n}_{L}_{N}_{seed}.json"
    with open(os.path.join(folder, name), "w") as final:
        json.dump(rho_values, final)

if __name__ == '__main__':

    monte_carlo_size = 1000
    bootstrap_size = 500
    seed = 7382197219

    true_alpha = np.array([1,1,1,1])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)
    #experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)

    true_alpha = np.array([3,1,1,3])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)
    #experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)

    true_alpha = np.array([1,7.4,2.6,1])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)
    experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)

    true_alpha = np.array([0.4,0.5,1.4,0.7])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)
    experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)

    true_alpha = np.array([19,1,1,19])
    #experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)
    experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)


    #n = 50
    #mu = np.array([0,0])
    #sigma = np.array([[1.0, 0.1], [0.1, 1.0]])
    #experiment_logitnormal(mu, sigma, n, monte_carlo_size, seed)

    #mu = np.array([-1.0, -1.0])
    #sigma = np.array([[2.25, -1.2], [-1.2, 1]])
    #experiment_logitnormal(mu, sigma, n, monte_carlo_size, seed)

    # SBC experiments

    #sbc_a = 1
    # b = 1
    # c = 0
    #sbc_s_mu = 1
    #sbc_s_sd = 0.5
    #sbc_n = 50
    #sbc_L = 63
    #sbc_N = 1000
    #sbc_seed = 831290
    # simulated_based_calibration(sbc_a, b, c, sbc_n, sbc_L, sbc_N, sbc_seed)
    #simulated_based_calibration_lognormal(sbc_a, sbc_s_mu, sbc_s_sd, sbc_n, sbc_L, sbc_N, sbc_seed)

    # Settin (c), true_alpha = [1.0, 7.4, 2.6, 1.0], with varying priors
    #monte_carlo_size_new = 1000
    #test_alpha = np.array([1.0, 7.4, 2.6, 1.0])
    #seed = 37812984
      
    # s ~ Gamma(4,1), \theta ~ Dirichlet(1,1,1,1) using bivariate-beta-model-gamma.stan 
    #experiment_bivbeta(test_alpha, 50, monte_carlo_size_new, bootstrap_size, seed=seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1)

    # s ~ Gamma(4,1/3), \theta ~ Dirichlet(1,1,1,1) using bivariate-beta-model-gamma.stan 
    #experiment_bivbeta(test_alpha, 50, monte_carlo_size_new, bootstrap_size, seed=seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 1, 1, 1]), prior_kappa=4, prior_lambda=1/3)

    # s ~ Gamma(10,5/6), \theta ~ Dirichlet(1,6,2,1) using bivariate-beta-model-gamma.stan 
    #experiment_bivbeta(test_alpha, 50, monte_carlo_size_new, bootstrap_size, seed=seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 6, 2, 1]), prior_kappa=10, prior_lambda=5/6)

    # s ~ Gamma(12,1), \theta ~ Dirichlet(1,7.4,2.6,1) using bivariate-beta-model-gamma.stan 
    #experiment_bivbeta(test_alpha, 50, monte_carlo_size_new, bootstrap_size, seed=seed, stan_model='bivariate-beta-model-gamma.stan', prior_a=np.array([1, 7.4, 2.6, 1]), prior_kappa=12, prior_lambda=1)
