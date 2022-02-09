#usr/bin/env python3
"""Figure generator 

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It plots the figures used 
in the paper.

This script requires that `numpy` and `matplotlib` be installed within the Python 
environment you are running. images
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm
from parameter_estimation import BivariateBeta
from __init__ import ROOT_DIR

plt.style.use('ggplot')

def plotting_bivariate_beta_pdf(alphas, n_points = 100):
    """
    Plots a grid of 3 x 4 densities of the bivariate beta distribution 
    with parameter alpha.
    
    Parameters:
    | alphas (12 x 4 - array): each line represents a different parameter specification.
    | n_points (int): numper of points to devide each component of the grid.
    """
    x_values = y_values = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X)

    fig = plt.figure(figsize = (16, 12))

    for k, alpha in tqdm(enumerate(alphas)):
        distribution = BivariateBeta(alpha=alpha)
        for i,x in enumerate(x_values): 
            for j,y in enumerate(y_values): 
                Z[i,j] = distribution.pdf(x, y)
    
        ax = fig.add_subplot(3, 4, k+1, projection='3d') 
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       cmap='magma', edgecolor='none')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(r"$\alpha =$ {}".format(alpha))
    plt.savefig(os.path.join(ROOT_DIR, '../figures/joint-densities-bivariate-beta.pdf'), bbox_inches = 'tight')

def plotting_positive_sets(v1_list, rho_list, n_points = 1000): 
    """
    Plots the sets where the solution to the system is strictly positive.
    Parameters:
    | v1_list (list): list of values for v1
    | rho_list (list): list of values for rho
    """ 
    distribution = BivariateBeta()
    def z_function(m1, m2, v1, rho):
        return 1*(sum(distribution._system_solution(m1, m2, v1, rho) <= 0) == 0) - 1*(v1 >= m1 - m1*m1)
    m1 = np.linspace(0, 1, n_points, endpoint=False)
    m2 = np.linspace(0, 1, n_points, endpoint=False)

    fig, ax = plt.subplots(3,5,figsize = (16,7), sharex = True, sharey = True)
    for k, (v1, rho) in enumerate(zip(v1_list, rho_list)): 
        i = k // 5
        j = k % 5
        x, y = np.meshgrid(m1,m2)
        data = z_function(x, y, v1, rho)
        ax[i,j].contourf(x, y, data, levels = [-1, -0.1, 0.5, 1], colors = ['grey', '#E08F4C', 'midnightblue'])
        ax[i,j].set_title(r'$v_1$ = {} and $\rho$ = {}'.format(v1, rho), fontsize = 12)
        
    for i in range(3): 
        ax[2,i].set_xlabel(r'$m_1$', fontsize = 15)
        ax[i,0].set_ylabel(r'$m_2$', fontsize = 15)
    ax[2,3].set_xlabel(r'$m_1$', fontsize = 15)
    ax[2,4].set_xlabel(r'$m_1$', fontsize = 15)
        
    fig.legend(handles = [mpatches.Patch(color='grey', label=r'$v_1 > m_1 - m_1^2$'.format(k+1)),
                          mpatches.Patch(color='#E08F4C', label=r'$\alpha_i < 0$ for some $i$'.format(k+1)),
                          mpatches.Patch(color='midnightblue', label=r'$\alpha_1, \dots, \alpha_4 > 0$'.format(k+1))], 
               loc = 'lower right', fontsize=12)
    plt.savefig(os.path.join(ROOT_DIR, '../figures/alpha_solution_existence.pdf'), bbox_inches = 'tight')

if __name__ == '__main__':

    f = input("Do you want to plot the densities?[y/n]")
    if f == 'y':
        alphas = np.array([[1,1,1,1],
                        [10,10,10,10],
                        [.5,.5,.5,.5],
                        [.25,.25,.25,.25],
                        [8,2,2,2],
                        [2,8,2,2],
                        [2,2,8,2],
                        [2,2,2,8],
                        [5,2,5,1],
                        [8,1,1,8],
                        [1,8,8,1],
                        [1,1,1,0.25],
                        ])
        plotting_bivariate_beta_pdf(alphas)
        plt.show()

    f = input("Do you want to plot the sets of positivity?[y/n]")
    if f == 'y':
        plotting_positive_sets([0.0001]*5 + [0.05]*5 + [0.1]*5, 
                               [-0.8, -0.3, 0, 0.3, 0.8]*4)
        plt.show()