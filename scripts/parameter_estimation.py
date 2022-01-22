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
from scipy.optimize import minimize, minimize_scalar

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
        alpha_sum = (m1 - m1*m1)/v1 - 1
        alpha_4 = alpha_sum * (rho * np.sqrt(m1 * m2 * (1 - m1) * (1 - m2)) + (1 - m1) * (1 - m2))
        alpha_1 = (m1 + m2 - 1) * alpha_sum + alpha_4
        alpha_2 = (1 - m2) * alpha_sum - alpha_4
        alpha_3 = (1 - m1) * alpha_sum - alpha_4
        return np.array([alpha_1, alpha_2, alpha_3, alpha_4])

    def _system_three_solution(self, m1, m2, rho):
        sqrt = rho * np.sqrt(m1 * m2 * (1-m1) * (1-m2))
        denominator = (1-m1) * (1-m2) + sqrt
        alpha1 = (m1 * m2 + sqrt) / denominator
        alpha2 = (m1 * (1-m2) - sqrt) / denominator
        alpha3 = (m2 * (1-m1) - sqrt) / denominator
        return np.array([alpha1, alpha2, alpha3, 1])

    def _system_two_solution(self, m1, m2, alpha3, alpha4):
        alpha1 = ((m1 + m2 - 1) * alpha3 + m2 * alpha4) / (1 - m1)
        alpha2 = ((1 - m2) * alpha3 + (m1 - m2) * alpha4) / (1 - m1)
        return (alpha1, alpha2)

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

    def loss_function(self, alpha, m1, m2, v1, v2, rho, g, c=np.ones(5)):
        """
        Calculates the loss function
        c[0]*g(m1,E[X1])+c[1]*g(m2,E[X2])+c[2]*g(v1,Var[X1])+c[3]*g(v2,Var[X2])+c[4]*g(rho,Cor(X1,X2))
        as function of alpha, when the means (m1,m2), variances (v1,v2), and 
        correlatation is fixed.
        Parameters:
        | alpha (4-array): parameter to calculate the loss
        | m1 (float): sample mean of the first component. 
        | m2 (float): sample mean of the second component. 
        | v1 (float): sample variance of the first component. 
        | v2 (float): sample variance of the second component. 
        | rho (float): sample correlation between the components. 
        | g (callable): function that receives to real values and return one. 
        | c (4-array): weights. Default: [1,1,1,1,1]

        Returns
        | loss (float) value representing the loss in the choice of this alpha.
        """
        alpha_sum = sum(alpha)
        div = alpha_sum*alpha_sum*(alpha_sum + 1)     
        alpha_12 = alpha[0] + alpha[1]
        alpha_34 = alpha[2] + alpha[3]
        alpha_13 = alpha[0] + alpha[2]
        alpha_24 = alpha[1] + alpha[3]
        loss = c[0]*g(m1, alpha_12/alpha_sum)
        loss += c[1]*g(m2, alpha_13/alpha_sum)
        loss += c[2]*g(v1, alpha_12*alpha_34/div)
        loss += c[3]*g(v2, alpha_13*alpha_24/div)
        loss += c[4]*g(rho, (alpha[0]*alpha[3] - alpha[1]*alpha[2])/(np.sqrt(alpha_12*alpha_34*alpha_13*alpha_24)))
        return loss

    def _choose_loss_function(self, code='l2'):
        """
        There are the following implemented loss functions: 
        - squared (code='l2') - Default
        - absolute (code='l1')
        - relative quadratic (code='rq')
        - relative absoltute (code='ra')
        """
        if code == 'l2':
            return lambda x,y: (x-y)*(x-y)
        elif code == 'l1':
            return lambda x,y: abs(x-y)
        elif code == 'rq':
            return lambda x,y: (x-y)*(x-y)/(x*x)
        elif code == 'ra':
            return lambda x,y: abs((x-y)/x)
        else:
            raise Exception('This loss function is not implemented. Please, implement it and let it as parameter.')

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
        denominator = np.sqrt(m1*m2*(1-m1)*(1-m2))

        if not accept_zero:
            if rho*denominator < -min(m1*m2, (1-m1)*(1-m2)) or rho*denominator > min(m1, m2) - m1*m2:
                return np.ones(4) * np.nan

        alpha_hat = self._system_three_solution(m1, m2, rho)
        alpha_sum = lambda alpha4: alpha4/((1-m1)*(1-m2) + rho*denominator)

        func_to_min = lambda alpha4: (v1 - m1*(1-m1)/(alpha_sum(alpha4) + 1))**2 + (v2 - m2*(1-m2)/(alpha_sum(alpha4) + 1))**2
        result = minimize_scalar(fun=func_to_min, bounds=(0, np.inf))
        alpha_hat = alpha_hat * result.x
        return np.maximum(alpha_hat, 0)

    def method_moments_estimator_3(self, x, y, alpha0):
        """
        Method of moments estimator of parameter alpha given the bivariate data (x,y) of size n.
        This method (MM3) solves the system with three euations (m1 and m2) and chooses alpha3 and alpha4 
        as the value to minimize the quadratic difference to the variances and correlation
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | alpha0 (2-array): initial guess for the optimizer (alpha3_0, alpha4_0)

        Returns: 
        | alpha_hat: estimator
        """
        m1 = np.mean(x)
        m2 = np.mean(y)
        v1 = np.var(x, ddof=1)
        v2 = np.var(y, ddof=1)
        rho = np.corrcoef(x, y)[0,1]

        def func_to_min(alphas):
            alpha3, alpha4 = tuple(alphas)
            alpha1, alpha2 = self._system_two_solution(m1, m2, alpha3, alpha4)
            alpha_sum = (alpha3 + alpha4) / (1 - m1)
            loss = (v1 - m1*(1-m1)/(alpha_sum + 1))**2 
            loss += (v2 - m2*(1-m2)/(alpha_sum + 1))**2
            loss += (rho - (alpha1 * alpha4 - alpha2 * alpha3)/(alpha_sum**2 * np.sqrt(m1*m2*(1-m1)*(1-m2))))**2
            return loss
                
        result = minimize(fun=func_to_min, bounds=[(0, np.inf)]*2, x0=alpha0, 
                         constraints=[{'type': 'ineq', 'fun': lambda x: (m1+m2-1)*x[0] + m2*x[1]}, 
                                      {'type': 'ineq', 'fun': lambda x: (1-m2)*x[0] + (m1-m2)*x[1]}], 
                         method='trust-constr',
                         options={'xtol': 1e-10, 'gtol': 1e-10})
        alpha_hat = np.ones(4)
        alpha_hat[2:] = result.x
        alpha_hat[:2] = self._system_two_solution(m1, m2, alpha_hat[2], alpha_hat[3])
        return alpha_hat

    def method_moments_estimator_4(self, x, y, alpha0, g=None, c=np.ones(5)):
        """
        Method of moments estimator of parameter alpha given the bivariate data (x,y) of size n.
        This method (MM4) searches the best alpha minimizing the quadratic differences with respect to
        m1, m2, v1, v2, and rho.
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | alpha0 (4-array): initial guess for the optimizer (alpha1_0, alpha2_0, alpha3_0, alpha4_0)
        | g (callable or str): function that receives to real values and return one.  Default: squared error
        | c (4-array): weights. Default: [1,1,1,1,1]

        Returns: 
        | alpha_hat: estimator
        """
        m1 = np.mean(x)
        m2 = np.mean(y)
        v1 = np.var(x, ddof=1)
        v2 = np.var(y, ddof=1)
        rho = np.corrcoef(x, y)[0,1]

        if g is None:
            g = self._choose_loss_function()
        elif isinstance(g, str):
            g = self._choose_loss_function(code=g)
        elif callable(g):
            pass
        else:
            raise Exception('g should be a callable, None or string.')

        result = minimize(fun=self.loss_function, 
                          x0=alpha0,
                          args=(m1, m2, v1, v2, rho, g, c),
                          bounds=[(0, np.inf)]*4,
                          constraints={'type': 'ineq', 
                                        'fun': lambda alpha: max(m1*(1-m1)/v1-1, m2*(1-m2)/v2-1) - sum(alpha)},
                          method='trust-constr',
                          options={'xtol': 1e-10, 'gtol': 1e-10})
        alpha_hat = result.x
        return alpha_hat

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

    np.random.seed(738912)
    true_alpha = np.array([0.3,4,3,5])
    U = np.random.dirichlet(true_alpha, size=1000000)
    X = U[:, 0] + U[:, 1]
    Y = U[:, 0] + U[:, 2]
    distribution = BivariateBeta()
    alpha_hat = distribution.method_moments_estimator_1(X, Y)
    print(alpha_hat)
    alpha_hat = distribution.method_moments_estimator_2(X, Y)
    print(alpha_hat)
    alpha_hat = distribution.method_moments_estimator_3(X, Y, alpha0=(1, 1))
    print(alpha_hat)
    alpha_hat = distribution.method_moments_estimator_4(X, Y, alpha0=(1, 1, 1, 1))
    print(alpha_hat)