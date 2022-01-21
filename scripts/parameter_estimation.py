#usr/bin/env python3
"""Parameter estimation of Bivariate Beta form Olkin and Trikalinos (2015)

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It allows the user to estimate 
the parameter alpha as explained in the notes. 

This script requires that `numpy` and `scipy` be installed within the Python 
environment you are running. 
"""
import numpy as np
from scipy.special import gamma, loggamma
from scipy.integrate import quad
from scipy.optimize import minimize

class BivariateBeta:
    """
    Perform parameter estimation and other functions related to 
    the Bivariate Beta distribution.
    """
    def __init__(self, alpha = None) -> None:
        """
        Initializes class variables. 
        Parameters:
        | alpha (4-array): parameter of the bivariate beta. If not passed, it is assumed to be [1,1,1,1].
        """
        self.alpha = np.array(alpha)
        if self.alpha is None: self.alpha = np.ones(4)

    def _integral_pdf(self, u, x, y, alpha):
        if (u == 0) or (u == x) or (u == y) or (u == x+y-1): 
            return 0
        fun  = u**(alpha[0]-1)
        fun *= (x-u)**(alpha[1]-1)
        fun *= (y-u)**(alpha[2]-1)
        fun *= (1-x-y+u)**(alpha[3]-1)
        return fun

    def pdf(self, x, y) -> float:
        """
        Returns the pdf value for given x and y, and a parameter alpha pre-specified.

        Parameters:
        | x (float): a value between 0 and 1
        | y (float): a value between 0 and 1

        Returns:
        | result (float): density of bivariate beta distribution at (x,y)
        """
        # verifications
        if x <= 0 or x >= 1 or y <= 0 or y >= 1: return 0.0

        # convergence problems
        if self.alpha[0] + self.alpha[3] <= 1: 
            if abs(x + y - 1) <= 1e-7: return 0.0
        if self.alpha[1] + self.alpha[2] <= 1:
            if abs(x - y) <= 1e-7: return 0.0
        if x <= 1e-7 or y <= 1e-7: return 0.0
        if 1-x <= 1e-7 or 1-y <= 1e-7: return 0.0
        
        # normalizing constant
        c = gamma(self.alpha).prod()/gamma(self.alpha.sum())
        
        lb = max(0,x+y-1)
        ub = min(x,y)
        result = quad(self._integral_pdf, lb, ub, args = (x, y, self.alpha), epsabs=1e-10, limit=50)[0]
        return result/c

    def log_pdf(self, x, y, alpha = None):
        """
        Returns the log pdf value for given x and y, and a parameter alpha pre-specified.

        Parameters:
        | x (float): a value between 0 and 1
        | y (float): a value between 0 and 1

        Returns:
        | result (float): density of bivariate beta distribution at (x,y)
        """
        if alpha is None: alpha = self.alpha

        if x <= 0 or x >= 1 or y <= 0 or y >= 1: return -np.inf
        # convergence problems
        if alpha[0] + alpha[3] <= 1:
            if abs(x + y - 1) <= 1e-7: return -np.inf
        if alpha[1] + alpha[2] <= 1:
            if abs(x - y) <= 1e-7: return -np.inf
        if x <= 1e-7 or y <= 1e-7: return -np.inf
        if 1-x <= 1e-7 or 1-y <= 1e-7: return -np.inf

        lb = max(0,x+y-1)
        ub = min(x,y)
        result = quad(self._integral_pdf, lb, ub, args = (x, y, alpha), epsabs=1e-10, limit=100)[0]
        if result == 0: return -np.inf
        result = np.log(result)
        return result

    def moments(self) -> np.array:
        """
        Calculates the main moments of the bivariate beta distribution: 
        E[X], E[Y], Var(X), Var(Y) and Corr(X,Y).

        Returns:
        | result (5-array): moments of the distribution 
        """
        alpha = self.alpha
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

    def _system_solution(self, m1, m2, v1, rho):
        alpha_bar = (m1 - m1*m1)/v1 - 1
        alpha_4 = rho * alpha_bar * np.sqrt(m1 * m2 * (1 - m1) * (1 - m2)) + (1 - m1) * (1 - m2)
        alpha_1 = (m1 + m2 - 1) * alpha_bar + alpha_4
        alpha_2 = (1 - m2) * alpha_bar - alpha_4
        alpha_3 = (1 - m1) * alpha_bar - alpha_4
        return np.array([alpha_1, alpha_2, alpha_3, alpha_4])

    def _system_three_solution(self, m1, m2, rho):
        sqrt = rho * np.sqrt(m1 * m2 * (1-m1) * (1-m2))
        denominator = (1-m1) * (1-m2) + sqrt
        alpha1 = (m1 * m2 + sqrt) / denominator
        alpha2 = (m1 * (1-m2) - sqrt) / denominator
        alpha3 = (m1 * (1-m1) - sqrt) / denominator
        return np.array([alpha1, alpha2, alpha3, 1])

    def system_solver(self, m1, m2, v1, v2, rho, check_v2=True):
        """
        Solves the system of equations built by the method of moments 
        including both means, one variance and the correlation. If the second variance 
        is more important, please swap the order of the variables.
        Parameters:
        | m1 (float): sample mean of the first component. 
        | m2 (float): sample mean of the second component. 
        | v1 (float): sample variance of the first component. 
        | v2 (float): sample variance of the second component. 
        | rho (float): sample correlation between the components. 

        Returns: 
        | alpha_hat (4-array): estimator of moments.
        """
        # verifications 
        if m1 <= 0 or m1 >= 1 or m2 <= 0 or m2 >= 1:
            raise Exception("The means should be between 0 and 1.")
        if v1 >= m1 - m1 * m1 or v2 >= m2 - m2 * m2:
            raise Exception("The variances should be in the correct interval.")
        if rho <= -1 or rho >= 1:
            raise Exception("The correlation must be in the open interval (-1,1).")
        if check_v2:
            if m1 * (1 - m1) * v2 !=  m2 * (1 - m2) * v1:
                print("The system has no solution. Excluding the equation regarding v2.")

        alpha_hat = self._system_solution(m1, m2, v1, rho)
        if sum(alpha_hat <= 0) > 0:
            print("The system has a non-positive solution.")
        return alpha_hat

    def method_moments_estimator_1(self, x, y, accept_zero=True):
        """
        Method of moments estimator of parameter alpha given the bivariate data (x,y) of size n.
        This method (MM1) solves the system and returns 0 whenever the solution is negative.
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | accept_zero (boolean): if True, accepts a zero as estimate. If False, the system return 
                                 nan values for the estimates if one of them is negative.

        Returns: 
        | alpha_hat: estimator
        """
        m1 = np.mean(x)
        m2 = np.mean(y)
        v1 = np.var(x, ddof=1)
        rho = np.corrcoef(x, y)[0,1]
        alpha_hat = self._system_solution(m1, m2, v1, rho)
        if not accept_zero:
            if sum(alpha_hat <= 0) > 0:
                return np.ones(4) * np.nan
        return np.maximum(alpha_hat, 0)

    def method_moments_estimator_2(self, x, y, accept_zero=True):
        """
        Method of moments estimator of parameter alpha given the bivariate data (x,y) of size n.
        This method (MM2) solves the system with three euations (m1, m2 and rho) and chooses alpha4 
        as the value to minimize the quadratic difference to the variances.
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | accept_zero (boolean): if True, accepts a zero as estimate. If False, the system return 
                                 nan values for the estimates if one of them is negative.

        Returns: 
        | alpha_hat: estimator
        """
        m1 = np.mean(x)
        m2 = np.mean(y)
        v1 = np.var(x, ddof=1)
        v2 = np.var(y, ddof=1)
        rho = np.corrcoef(x, y)[0,1]
        alpha_hat = self._system_three_solution(m1, m2, rho)

        if not accept_zero:
            if sum(alpha_hat <= 0) > 0:
                return np.ones(4) * np.nan
        return np.maximum(alpha_hat, 0)

    def maximum_likelihood_estimator(self, x, y, alpha0):
        """
        Maximum likelihood estimator of parameter alpha given the bivariate data (x,y) of size n.
        Parameters 
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | alpha0 (4-array): initial guess for the parameters

        Returns: 
        | alpha_hat: mle
        """
        def likelihood(alpha, x, y):
            log_pdf = sum([self.log_pdf(i, j, alpha) for (i, j) in zip(x,y)])
            # normalizing constant
            c = loggamma(alpha).sum() - loggamma(alpha.sum())
            L_neg = -(log_pdf - len(x) * c)
            return L_neg
            
        res = minimize(fun=likelihood, x0=alpha0, method='L-BFGS-B', args = (x,y), 
                       bounds=[(1e-5, 100)]*4, options={'eps': 1e-16})
        print(res)
        alpha_hat = res.x
        return alpha_hat
        
if __name__ == '__main__':

    np.random.seed(7821)    
    U = np.random.dirichlet([1,1,1,1], size=1000)
    X = U[:, 0] + U[:, 1]
    Y = U[:, 0] + U[:, 2]
    distribution = BivariateBeta()
    alpha_hat = distribution.maximum_likelihood_estimator(X, Y, alpha0 = (1,1,1,1))
    print(alpha_hat)