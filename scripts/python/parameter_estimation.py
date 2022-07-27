#usr/bin/env python3
"""Parameter estimation of Bivariate Beta form Olkin and Trikalinos (2015)

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It allows the user to estimate 
the parameter alpha as explained in the notes. 

This script requires that `numpy`, `scipy` and `lintegrate` be installed within the Python 
environment you are running. 
"""
import numpy as np
from scipy.special import gamma, loggamma, digamma, beta, hyp2f1
from mpmath import appellf1
from scipy.integrate import quad
from scipy.optimize import minimize, root
from functools import partial
import multiprocessing

# import matplotlib.pyplot as plt
from tqdm import tqdm
# import lintegrate


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

    def _integral_pdf(self, u, x, y, alpha, check = True):
        if check:
            if (u == 0) or (u == x) or (u == y) or (u == x+y-1): 
                return 0
        fun  = u**(alpha[0]-1)
        fun *= (x-u)**(alpha[1]-1)
        fun *= (y-u)**(alpha[2]-1)
        fun *= (1-x-y+u)**(alpha[3]-1)
        return fun

    def _log_integral_pdf(self, u, args):
        x, y, alpha = args
        fun = (alpha[0]-1)*np.log(u)
        fun += (alpha[1]-1)*np.log(x-u)
        fun += (alpha[2]-1)*np.log(y-u)
        fun += (alpha[3]-1)*np.log1p(-x-y+u)
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
        result = quad(self._integral_pdf, lb, ub, args = (x, y, self.alpha), epsabs=1e-10, limit=80)[0]
        return result/c

    def pdf_appell(self, x, y) -> float:
        """
        Returns the pdf value for given x and y, and a parameter alpha pre-specified. 
        This implementation use the hypergeometric formulation.

        Parameters:
        | x (float): a value between 0 and 1
        | y (float): a value between 0 and 1

        Returns:
        | result (float): density of bivariate beta distribution at (x,y)
        """
        if x <= 0 or x >= 1 or y <= 0 or y >= 1: return 0.0
        alpha1, alpha2, alpha3, alpha4 = tuple(self.alpha)

        # supposition of continuity
        if x+y == 1:
            y=1-x+1e-10

        # normalizing constant
        c = gamma(self.alpha).prod()/gamma(self.alpha.sum())

        try:
            if x+y<1:
                if x<y:
                    v = beta(alpha1, alpha2) * x**(alpha1+alpha2-1) * y**(alpha3-1) * (1-x-y)**(alpha4-1)
                    v *= appellf1(alpha1, 1-alpha3, 1-alpha4, alpha1+alpha2, x/y, x/(x+y-1), maxterms=5000)
                else:
                    v = beta(alpha1, alpha3) * x**(alpha2-1) * y**(alpha1+alpha3-1) * (1-x-y)**(alpha4-1)
                    v *= appellf1(alpha1, 1-alpha2, 1-alpha4, alpha1+alpha3, y/x, y/(x+y-1), maxterms=5000)
            else:
                if x<y:
                    v = beta(alpha2, alpha4) * (1-x)**(alpha3-1) * (1-y)**(alpha2+alpha4-1) * (x+y-1)**(alpha1-1)
                    v *= appellf1(alpha4, 1-alpha1, 1-alpha3, alpha2+alpha4, (1-y)/(1-x-y), (1-y)/(1-x), maxterms=5000)
                else:
                    v = beta(alpha3, alpha4) * (1-x)**(alpha3+alpha4-1) * (1-y)**(alpha2-1) * (x+y-1)**(alpha1-1)
                    v *= appellf1(alpha4, 1-alpha1, 1-alpha2, alpha3+alpha4, (1-x)/(1-x-y), (1-x)/(1-y), maxterms=5000)
        except ValueError:
            v = self._analytic_continuation(min(x,y), alpha1, alpha2, alpha3, alpha4)
        return float(v)/c

    def _analytic_continuation(self, x, alpha1, alpha2, alpha3, alpha4):
        """
        Analytic continuation implementation for x=y.
        """
        if alpha2+alpha3 < 1:
            raise Exception('The 2F1 function is not defined in this case as an integral.')
        if x < 1/2:
            v = beta(alpha1, alpha2+alpha3-1) * x**(alpha1+alpha2+alpha3-2) * (1-2*x)**(alpha4-1)
            v *= hyp2f1(1-alpha4, alpha1, alpha1+alpha2+alpha3-1,x/(2*x-1))
        else:
            v = beta(alpha4, alpha2+alpha3-1) * (1-x)**(alpha2+alpha3+alpha4-2) * (2*x-1)**(alpha1-1)
            v *= hyp2f1(1-alpha1, alpha4, alpha2+alpha3+alpha4-1,(x-1)/(2*x-1))
        return v

    def log_pdf(self, x, y, alpha = None, lb = 0, ub = 1):
        """
        Returns the log pdf value for given x and y, and a parameter alpha pre-specified.

        Parameters:
        | x (float): a value between 0 and 1
        | y (float): a value between 0 and 1

        Returns:
        | result (float): density of bivariate beta distribution at (x,y)
        """
        if alpha is None: alpha = self.alpha

        # convergence problems
        if alpha[0] + alpha[3] <= 1:
            if abs(x + y - 1) <= 1e-7: return -np.inf
        if alpha[1] + alpha[2] <= 1:
            if abs(x - y) <= 1e-7: return -np.inf
        if x <= 1e-7 or y <= 1e-7: return -np.inf
        if 1-x <= 1e-7 or 1-y <= 1e-7: return -np.inf

        # Uses the traditional quad function from scipyalternativo
        result = np.log(quad(self._integral_pdf, lb, ub, args = (x, y, alpha), epsabs=1e-10, limit=50)[0])
        # Uses the library lintegrate from mattpitkin
        #result = lintegrate.lqag(self._log_integral_pdf, lb, ub, args=(x,y,alpha))[0]
        return result

    def neg_log_likelihood(self, alpha, x, y, lb, ub):
        #self.beta_samples = np.random.beta(a=alpha[0], b=alpha[1], size=10000)
        log_pdf = sum([self.log_pdf(x_i, y_i, alpha, lb_i, ub_i) for (x_i, y_i, lb_i, ub_i) in zip(x,y,lb,ub)])
        # normalizing constant
        c = loggamma(alpha).sum() - loggamma(alpha.sum())
        L_neg = -(log_pdf - len(x) * c)
        return L_neg

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
        """
        This system solves for (alpha1/alpha4, alpha2/alpha4, alpha3/alpha4, 1). After choosing alpha4,
        you can multiply this vector to alpha4 and obtain alpha.
        """
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
        v = -0.5 * (np.log(alpha_12)+np.log(alpha_34)+np.log(alpha_13)+np.log(alpha_24))
        loss += c[4]*g(rho, (alpha[0]*alpha[3] - alpha[1]*alpha[2]) * np.exp(v))
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
        This method (MM2) solves the system with three equations (m1, m2 and rho) and chooses alpha4 
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
        # minimizing the sum of squares comparing the sum of alphas and its model value.
        alpha4 = ((1-m1)*(1-m2) + rho*denominator) * ((m1*(1-m1)/v1 + m2*(1-m2)/v2)/2 - 1)
        alpha_hat = alpha4 * alpha_hat
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
        E = m2*(1-m1) - rho * np.sqrt(m1*m2*(1-m1)*(1-m2))

        def func_to_min(alphas, E):
            alpha3, alpha4 = tuple(alphas)
            alpha_sum = (alpha3 + alpha4) / (1 - m1)
            loss = (alpha_sum - m1*(1-m1)/v1 + 1)**2
            loss += (alpha_sum - m2*(1-m2)/v2 + 1)**2
            loss += (E * alpha_sum - alpha3)**2
            return loss
        
        def derivative(alphas, E):
            alpha3, alpha4 = tuple(alphas)
            alpha_sum = (alpha3 + alpha4) / (1-m1)
            grad_alpha3 = (2*alpha_sum - m1*(1-m1)/v1 - m2*(1-m2)/v2 + 2) + (E * alpha_sum - alpha3) * E
            grad_alpha4 = grad_alpha3
            grad_alpha3 -= (E * alpha_sum - alpha3) * (1-m1)
            return 2*np.array([grad_alpha3, grad_alpha4])/(1-m1)
            
        result = minimize(fun=func_to_min, bounds=[(0, np.inf)]*2, x0=alpha0,
                         constraints=[{'type': 'ineq', 'fun': lambda x: (m1+m2-1)*x[0] + m2*x[1]}, 
                                      {'type': 'ineq', 'fun': lambda x: (1-m2)*x[0] + (m1-m2)*x[1]}],
                         args = (E,),
                         jac=derivative,
                         method='SLSQP',
                         options={'ftol': 1e-10})
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
                                        'fun': lambda alpha: max(m1*(1-m1)/v1, m2*(1-m2)/v2) - 1 - sum(alpha)},
                          method='SLSQP',
                          options={'ftol': 1e-10})
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
        #self.uniform = np.random.uniform(size=10000)
        lower_bounds = np.maximum(0, x+y-1)
        upper_bounds = np.minimum(x, y)
        res = minimize(fun=self.neg_log_likelihood, x0=alpha0, method='trust-constr', 
                       args = (x, y, lower_bounds, upper_bounds), 
                       bounds=[(0.1, 6)]*4, options={'gtol': 1e-16})
        alpha_hat = res.x
        return alpha_hat

    def modified_maximum_likelihood_estimator(self, x, y, x0=(2,2,4)):
        """
        Modified likelihood estimator of parameter alpha given the bivariate data (x,y) of size n.
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | x0 (3-array): initial guess to the root finder program

        Returns
        | alpha_hat: mle
        """
        rho = np.corrcoef(x, y)[0,1]

        def system_equations(parameters, x, y):
            a, b, c = tuple(parameters)
            dc = digamma(c)
            fun1 = np.log(x).mean() - digamma(a)
            fun2 = np.log1p(-x).mean() - digamma(c-a)
            fun3 = np.log(y).mean() - digamma(b)
            #fun4 = np.log1p(-y).mean() - digamma(c-b)
            return np.array([fun1, fun2, fun3]) + dc

        # def jacobian(parameters, x, y):
        #     a, b, c = tuple(parameters)
        #     n = len(x)
        #     dpsi_a = polygamma(n=1, x=a)
        #     dpsi_b = polygamma(n=1, x=b)
        #     dpsi_c = polygamma(n=1, x=c)
        #     dpsi_cb = polygamma(n=1, x=c-b)
        #     dpsi_ca = polygamma(n=1, x=c-a)
        #     return np.array([[-dpsi_a, 0, dpsi_c], 
        #                      [dpsi_ca, 0, dpsi_c - dpsi_ca],
        #                      [0, -dpsi_b, dpsi_c],
        #                      [0, dpsi_cb, dpsi_c - dpsi_cb]])

        sol = root(fun=system_equations, x0=x0, args=(x,y), method='hybr')
        if not sol.success:
            return np.zeros(4) * np.nan
        a, b, c = tuple(sol.x)

        alpha4_hat = (rho * np.sqrt(a*b*(c-a)*(c-b)) + (c-a)*(c-b))/c
        alpha4_hat = max(0, min(alpha4_hat, c - max(a,b)))
        alpha1_hat = max(0, a + b - c + alpha4_hat)
        alpha2_hat = c - b - alpha4_hat
        alpha3_hat = c - a - alpha4_hat
        alpha_hat = np.array([alpha1_hat, alpha2_hat, alpha3_hat, alpha4_hat])
        return alpha_hat

    def _bootstrap_wrapper(self, b, X, Y, alpha0, x0, method):
        """
        Maps the bootstrap sample for the methods wrapper.
        """
        return self._methods_wrapper(X[:, b], Y[:, b], alpha0=alpha0, x0=x0, method=method)

    def _methods_wrapper(self, x, y, alpha0, x0, method):
        """
        Maps the methods which demands different parameters.
        """
        if alpha0 is None and x0 is None:
            alpha_hat = method(x, y)
        elif x0 is None:
            alpha_hat = method(x, y, alpha0=alpha0)
        else:
            alpha_hat = method(x, y, x0=x0)
        return alpha_hat

    def bootstrap_method(self, x, y, B, method, processes=2, seed=1000, alpha0=None, x0=None):
        """
        Bootstrap samples for the estimated parameters alpha. It resamples with replacement 
        from x and y B times and for each estimate the parameter. 
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | B (int): number of bootstrap samples
        | method (fuction): a function that receives arrays x and y, and returns an alpha. Pass alpha0 if necessary.
        | seed (int): seed of the random object used in the function.

        Returns
        | boostrap_sample (4xB-array): estimated parameters for each resample.
        """
        ro = np.random.RandomState(seed)
        index = ro.choice(range(len(x)), size=(len(x), B))
        X = x[index]
        Y = y[index]
        pool = multiprocessing.Pool(processes=processes)
        estimating_b = partial(self._bootstrap_wrapper, X=X, Y=Y, alpha0=alpha0, x0=x0, method=method)
        bootstrap_sample = np.array(pool.map(estimating_b, range(B))).transpose()

        return bootstrap_sample

    def bootstrap_method_parametric(self, x, y, B, method, processes=2, seed=1000, alpha0=None, x0=None):
        """
        Bootstrap samples for the estimated parameters alpha. It resamples with replacement 
        from x and y B times and for each estimate the parameter. 
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | B (int): number of bootstrap samples
        | method (fuction): a function that receives arrays x and y, and returns an alpha. Pass alpha0 if necessary.
        | seed (int): seed of the random object used in the function.

        Returns
        | boostrap_sample (4xB-array): estimated parameters for each resample.
        """
        ro = np.random.RandomState(seed)
        alpha_hat = self._methods_wrapper(x, y, alpha0=alpha0, x0=x0, method=method)
        U = ro.dirichlet(alpha_hat, size=(len(x), B))
        X = U[:,:,0] + U[:,:,1]
        Y = U[:,:,0] + U[:,:,2]
        pool = multiprocessing.Pool(processes=processes)
        estimating_b = partial(self._bootstrap_wrapper, X=X, Y=Y, alpha0=alpha0, x0=x0, method=method)
        bootstrap_sample = np.array(pool.map(estimating_b, range(B))).transpose()

        return bootstrap_sample

    def confidence_interval(self, level, samples):
        """
        Calculate the percentile interval of level 'level' (for instance level .95).
        Parameters
        | level (float): number between 0 and 1
        | samples (n-array): the array to calculate the percentiles

        Returns
        | ci (2-array): array with the confidence interval 
        """
        ci = np.quantile(samples, q=[(1-level)/2, (1+level)/2], axis=1)
        return ci

if __name__ == '__main__':

    true_alpha = np.array([1,1,1,1])
    n = 50
    rho_samples = np.zeros(10000)
    for i in tqdm(range(rho_samples.shape[0])):
        U = np.random.dirichlet(true_alpha, size=n)
        X = U[:,0] + U[:,1]
        Y = U[:,0] + U[:,2]
        rho = np.corrcoef(X,Y)[0,1]
        rho_samples[i] = rho
    print(np.quantile(rho_samples, q=[0.05, 0.9]))

    n = 50
    rho_samples = np.zeros(10000)
    for i in tqdm(range(rho_samples.shape[0])):
        U = 1/(1 + np.exp(-np.random.multivariate_normal(mean=[0,0], cov=[[1,0.0], [0.0, 1]], size=n)))
        X = U[:,0]
        Y = U[:,1]
        rho = np.corrcoef(X,Y)[0,1]
        rho_samples[i] = rho
    print(np.quantile(rho_samples, q=[0.05, 0.9]))
