#usr/bin/env python3
"""Parameter estimation of Bivariate Beta form Olkin and Trikalinos (2015)

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It allows the user to estimate 
the parameter alpha as explained in the notes. 

This script requires that `numpy` be installed within the Python 
environment you are running. 
"""
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

class BivariateBeta:
    """
    Perform parameter estimation and other functions related to 
    the Bivariate Beta distribution.
    """
    def __init__(self) -> None:
        pass

    def _integral_pdf(self, u, x, y, alpha):
        if (u == 0) or (u == x) or (u == y) or (u == x+y-1): 
            return 0
        fun  = u**(alpha[0]-1)
        fun *= (x-u)**(alpha[1]-1)
        fun *= (y-u)**(alpha[2]-1)
        fun *= (1-x-y+u)**(alpha[3]-1)
        return fun

    def pdf(self, x, y, alpha) -> float:
        """
        Returns the pdf value for given x and y, and a parameter alpha pre-specified.

        Parameters:
        | x (float): a value between 0 and 1
        | y (float): a value between 0 and 1
        | alpha (4-array): parameter array with size 4 

        Returns:
        | result (float): density of bivariate beta distribution at (x,y)
        """
        # verifications
        alpha = np.array(alpha)
        if len(alpha) != 4: 
            raise Exception('The parameter must have size four.')
        if x <= 0 or x >= 1 or y <= 0 or y >= 1: return 0.0

        # convergence problems
        if alpha[0] + alpha[3] <= 1: 
            if abs(x + y - 1) <= 1e-7: return 0.0
        if alpha[1] + alpha[2] <= 1:
            if abs(x - y) <= 1e-7: return 0.0
        if x <= 1e-7 or y <= 1e-7: return 0.0
        if 1-x <= 1e-7 or 1-y <= 1e-7: return 0.0
        
        # normalizing constant
        c = gamma(alpha).prod()/gamma(alpha.sum())
        
        lb = max(0,x+y-1)
        ub = min(x,y)
        result = quad(self._integral_pdf, lb, ub, args = (x,y,alpha), epsabs=1e-10, limit=50)[0]
        return result/c

    def moments(self, alpha) -> np.array:
        """
        Calculates the main moments of the bivariate beta distribution: 
        E[X], E[Y], Var(X), Var(Y) and Corr(X,Y).

        Parameters:
        | alpha (4-array): parameter array with size 4

        Returns:
        | result (5-array): moments of the distribution 
        """
        alpha = np.array(alpha)
        alpha_sum = alpha.sum()

        E_X = (alpha[0] + alpha[1])/alpha_sum
        E_Y = (alpha[0] + alpha[2])/alpha_sum
        Var_X = (1/(alpha_sum * (alpha_sum+1))) * E_X * (alpha[2] + alpha[3])
        Var_Y = (1/(alpha_sum * (alpha_sum+1))) * E_Y * (alpha[1] + alpha[3])
        den = np.log(alpha[0] + alpha[1]) + np.log(alpha[2] + alpha[3]) 
        den += np.log(alpha[0] + alpha[2]) + np.log(alpha[1] + alpha[3])
        den = np.exp(-0.5*den)
        Cor_XY = (alpha[0]*alpha[3] - alpha[1]*alpha[2])*den
        result = np.array([E_X, E_Y, Var_X, Var_Y, Cor_XY])
        return result

if __name__ == '__main__':
    pass