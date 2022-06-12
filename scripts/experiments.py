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

import matplotlib.pyplot as plt
import numpy as np
from parameter_estimation import BivariateBeta
from tqdm import trange
from time import time
import os
import json
from __init__ import ROOT_DIR

def starting_experiment(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed):
    """
    Prepares the experiment file for the well-specified case, that is, the data comes from
    the Bivariate Beta distribution.
    """
    filename = '../experiments/exp_' + '_'.join(str(e) for e in true_alpha) 
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
    filename = '../experiments/exp_logit_' + '_'.join(str(e) for e in mu) + '_' + '_'.join(str(e) for e in sigma.flatten())
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
        t0 = time()
        alpha_hat5 = distribution.modified_maximum_likelihood_estimator(X, Y, x0=(2, 2, 4))
        time5 = time() - t0
        alpha = np.array([alpha_hat1, alpha_hat2, alpha_hat3, alpha_hat4, alpha_hat5])

        # Updating the estimates iteratively
        bias_new = alpha - true_alpha
        mse_new = bias_new * bias_new
        mape_new = abs(bias_new)/true_alpha
        comp_new = np.array([time1, time2, time3, time4, time5])

        if coverage:

            methods = [distribution.method_moments_estimator_1, distribution.method_moments_estimator_2, 
                       distribution.method_moments_estimator_3, distribution.method_moments_estimator_4, 
                       distribution.modified_maximum_likelihood_estimator]
            alpha0_parameters = [None, None, (1,1), (1,1,1,1), None]
            x0_parameters = [None, None, None, None, (2,2,4)]

            for ind in range(5):
                samples = distribution.bootstrap_method(x=X, y=Y, 
                                                        B=bootstrap_size,
                                                        method=methods[ind],
                                                        processes=4,
                                                        seed=rng.integers(2**32-1),
                                                        alpha0=alpha0_parameters[ind],
                                                        x0=x0_parameters[ind])
                ci = distribution.confidence_interval(level=0.95, samples=samples)
                coverage_new[ind, :] = (ci[0,:] < true_alpha)*(ci[1,:] > true_alpha)
        
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

    for exp in trange(monte_carlo_size):
        Z = rng.multivariate_normal(mu, sigma, size=sample_size)
        X = 1/(1 + np.exp(-Z[:, 0]))
        Y = 1/(1 + np.exp(-Z[:, 1]))
    
        alpha_hat1 = distribution.method_moments_estimator_1(X, Y)
        alpha_hat2 = distribution.method_moments_estimator_2(X, Y)
        alpha_hat3 = distribution.method_moments_estimator_3(X, Y, alpha0=(1, 1))
        alpha_hat4 = distribution.method_moments_estimator_4(X, Y, alpha0=(1, 1, 1, 1))
        alpha_hat5 = distribution.modified_maximum_likelihood_estimator(X, Y, x0=(2, 2, 4))

        est_moments1 = BivariateBeta(alpha=alpha_hat1).moments()
        est_moments2 = BivariateBeta(alpha=alpha_hat2).moments()
        est_moments3 = BivariateBeta(alpha=alpha_hat3).moments()
        est_moments4 = BivariateBeta(alpha=alpha_hat4).moments()
        est_moments5 = BivariateBeta(alpha=alpha_hat5).moments()
        est_moments = np.array([est_moments1, est_moments2, est_moments3, est_moments4, est_moments5])

        # Updating the estimates iteratively
        bias_new = est_moments - true_moments
        mse_new = bias_new * bias_new
        mape_new = abs(bias_new)/abs(true_moments)

        saving_document_2(filename, bias_new, mse_new, mape_new)

def variation_alpha4(true_alpha, sample_size, monte_carlo_size, seed):
    """
    It does the experiments from Section "Recovering parameters from bivariate beta".
    """
    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()

    moments_vs_alpha4 = np.zeros((monte_carlo_size, 6))

    for k in trange(monte_carlo_size):
        U = rng.dirichlet(true_alpha, size=sample_size)
        X = U[:, 0] + U[:, 1]
        Y = U[:, 0] + U[:, 2]

        alpha_hat1 = distribution.method_moments_estimator_1(X, Y)

        moments_vs_alpha4[k, 0] = X.mean()
        moments_vs_alpha4[k, 1] = Y.mean()
        moments_vs_alpha4[k, 2] = X.var(ddof=1)
        moments_vs_alpha4[k, 3] = Y.var(ddof=1)
        moments_vs_alpha4[k, 4] = np.corrcoef(X,Y)[0,1]
        moments_vs_alpha4[k, 5] = alpha_hat1[2]

    names = [r'$\hat{m}_1$', r'$\hat{m}_2$', r'$\hat{v}_1$', r'$\hat{v}_2$', r'$\hat{\rho}$']

    fig, ax = plt.subplots(1, 5, figsize=(20,4), sharey=True)
    fig.suptitle('Sensitivity analysis', fontsize=20)
    for i in range(5):
        ax[i].scatter(moments_vs_alpha4[:,i], moments_vs_alpha4[:,-1], s=1, color='black')
        ax[i].set_xlabel(names[i], fontsize=14)
    ax[0].set_ylabel(r'$\hat\alpha_4$', fontsize=14)

    plt.savefig(os.path.join(ROOT_DIR, '../figures/sensibility_analysis_alpha3.pdf'), bbox_inches='tight')
    plt.show()  

def comparing_methods(true_alpha, monte_carlo_size, bootstrap_size, seed):

    filename1 = starting_experiment(true_alpha, 50, monte_carlo_size, bootstrap_size, seed)
    filename2 = starting_experiment(true_alpha, 1000, monte_carlo_size, bootstrap_size, seed)
    with open(filename1, 'r') as f:
        experiment1 = json.load(f)
    with open(filename2, 'r') as f:
        experiment2 = json.load(f)

    methods = ['MM1', 'MM2', 'MM3', 'MM4']

    bias1 = [np.mean(np.abs(experiment1['bias'][i])) for i in range(4)]
    mape1 = [np.mean(experiment1['mape'][i]) for i in range(4)]
    bias2 = [np.mean(np.abs(experiment2['bias'][i])) for i in range(4)]
    mape2 = [np.mean(experiment2['mape'][i]) for i in range(4)]

    fig, ax = plt.subplots(2,2)

    ax[0,0].bar(methods, mape1, color='black')
    ax[0,1].bar(methods, bias1, color='black')
    ax[1,0].bar(methods, mape2, color='black')
    ax[1,1].bar(methods, bias2, color='black')

    ax[0,0].set_ylabel(r'$n=50$')
    ax[1,0].set_ylabel(r'$n=1000$')
    ax[0,0].set_title('Average MAPE')
    ax[0,1].set_title('Average absolute bias')

    ax[0,0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax[1,0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax[0,1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax[1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    fig.tight_layout() 
    plt.savefig(os.path.join(ROOT_DIR, '../figures/comparing_methods_mape_bias_XXX.pdf'), bbox_inches='tight')
    plt.show()

def comparing_methods2(mu1, mu2, sigma1, sigma2, sample_size, monte_carlo_size, seed):

    filename1 = starting_experiment_2(mu1, sigma1, sample_size, monte_carlo_size, seed)
    filename2 = starting_experiment_2(mu2, sigma2, sample_size, monte_carlo_size, seed)
    with open(filename1, 'r') as f:
        experiment1 = json.load(f)
    with open(filename2, 'r') as f:
        experiment2 = json.load(f)

    values = [r'$m_1$', r'$m_2$', r'$v_1$', r'$v_2$', r'$\rho$']
    mape1 = experiment1['mape'][3]
    mape2 = experiment2['mape'][3]

    fig, ax = plt.subplots(1,2)

    ax[0].bar(values, mape1, color='black')
    ax[1].bar(values, mape2, color='black')

    ax[0].set_ylabel('MAPE')
    ax[0].set_title('Experiment 1')
    ax[1].set_title('Experiment 2')

    # ax[0,0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    # ax[1,0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    # ax[0,1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    # ax[1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    fig.tight_layout() 
    #plt.savefig(os.path.join(ROOT_DIR, '../figures/comparing_methods_mape_bias_XXX.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    monte_carlo_size = 1000
    bootstrap_size = 500

    #true_alpha = np.array([1,1,1,1])
    #true_alpha = np.array([2,7,3,1])
    #true_alpha = np.array([0.7,0.9,2,1.5])
    #sample_size = 50
    #seed = 3781288
    #experiment_bivbeta(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed, coverage=False)

    #sample_size = 1000
    #experiment_bivbeta(true_alpha, sample_size, monte_carlo_size, bootstrap_size, seed, coverage=False)

    #comparing_methods(true_alpha, monte_carlo_size, bootstrap_size, seed)

    mu2 = np.array([-1, -1])
    sigma2 = np.array([[1, -0.8], [-0.8, 1]])
    mu1 = np.array([0, 0])
    sigma1 = np.array([[1, 0.1], [0.1, 1]])
    sample_size = 50
    seed = 63127371

    comparing_methods2(mu1, mu2, sigma1, sigma2, sample_size, monte_carlo_size, seed)

    #experiment_logitnormal(mu1, sigma1, sample_size, monte_carlo_size, seed)
    #experiment_logitnormal(mu2, sigma2, sample_size, monte_carlo_size, seed)

    # rng = np.random.default_rng(seed)
    # distribution = BivariateBeta()
    # Z = rng.multivariate_normal(mu, sigma, size=sample_size)
    # X = 1/(1 + np.exp(-Z[:, 0]))
    # Y = 1/(1 + np.exp(-Z[:, 1]))
    # print(X.mean(), Y.mean(), X.var(), Y.var(), np.corrcoef(X,Y)[0,1])

    # alpha_hat1 = distribution.method_moments_estimator_1(X, Y)
    # d = BivariateBeta(alpha=alpha_hat1)
    # print(d.moments())

    #true_alpha = np.array([2,4,3,1])
    #sample_size = 50
    #seed = 367219
    #variation_alpha4(true_alpha, sample_size, monte_carlo_size, seed)

    # rho_true = moments_logit_normal(mu, sigma)[-1]
    # rho_estimated = []
    # for i in trange(1000):
    #     Z = np.random.multivariate_normal(mu, sigma, size=sample_size)
    #     X = 1/(1 + np.exp(-Z[:, 0]))
    #     Y = 1/(1 + np.exp(-Z[:, 1]))
    #     estimated_moments = np.array([X.mean(), Y.mean(), X.var(ddof=1), Y.var(ddof=1), np.corrcoef(X,Y)[0,1]])
    #     rho_estimated.append(estimated_moments[-1])
    # #plt.hist(rho_estimated, color='black', bins=25)
    # print(np.mean(1*(np.array(rho_estimated) > 0.2) + 1*(np.array(rho_estimated) < 0)))
    # plt.show()
