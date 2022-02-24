#usr/bin/env python3
"""Parameter estimation of Bivariate Beta form Olkin and Trikalinos (2015)

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It allows the user to estimate 
the parameter alpha as explained in the notes. 

This script requires that `numpy` and `scipy` be installed within the Python 
environment you are running. 
"""
from time import time
from tqdm import tqdm
import numpy as np
from scipy.special import gamma, loggamma, digamma
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
import lintegrate

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
        result = quad(self._integral_pdf, lb, ub, args = (x, y, self.alpha), epsabs=1e-10, limit=50)[0]
        return result/c

    def _alternative_integral_pdf(self, x, y, alpha, lb, ub):
        u = self.uniform*(ub - lb) + lb
        return (ub - lb) * np.mean(self._integral_pdf(u, x, y, alpha, False))

    def _alternative_integral_pdf_2(self, x, y, alpha, lb, ub):
        lb, ub = lb/x, ub/x
        c = quad(func=lambda t, a, b: t**(a-1) * (1-t)**(b-1), 
                 a=lb, b=ub, args=(alpha[0], alpha[1]))[0]
        samples = self.beta_samples[(self.beta_samples>=lb)*(self.beta_samples<=ub)]
        monte_carlo = np.mean((y - x * samples)**(alpha[2]-1) * (1 - x - y + x * samples)**(alpha[3]-1))
        return c, monte_carlo

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
        # Uses Monte Carlo approximation with uniform draws
        #result = np.log(self._alternative_integral_pdf(x, y, alpha, lb, ub))
        # Uses Monte Carlo approximation with beta draws
        #result = self._alternative_integral_pdf_2(x, y, alpha, lb, ub)
        #result = np.log(result[0]) + (alpha[0]+alpha[1]-1)*np.log(x) + np.log(result[1])
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
                                        'fun': lambda alpha: max(m1*(1-m1)/v1, m2*(1-m2)/v2) - 1 - sum(alpha)},
                          method='trust-constr',
                          options={'xtol': 1e-10, 'gtol': 1e-10})
        alpha_hat = result.x
        return alpha_hat

    def neg_log_likelihood(self, alpha, x, y, lb, ub):
        #self.beta_samples = np.random.beta(a=alpha[0], b=alpha[1], size=10000)
        print(alpha)
        log_pdf = sum([self.log_pdf(x_i, y_i, alpha, lb_i, ub_i) for (x_i, y_i, lb_i, ub_i) in zip(x,y,lb,ub)])
        # normalizing constant
        c = loggamma(alpha).sum() - loggamma(alpha.sum())
        L_neg = -(log_pdf - len(x) * c)
        return L_neg

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
        print(res)
        alpha_hat = res.x
        return alpha_hat
        
    def modified_maximum_likelihood_estimator(self, x, y, alpha0):
        """
        Modified likelihood estimator of parameter alpha given the bivariate data (x,y) of size n.
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | 

        Returns
        | alpha_hat: mle
        """
        # TO BE FINISHED
        rho = np.corrcoef(x, y)[0,1]

        def system_equations(a, b, c, x, y):
            n = len(x)
            dc = digamma(c)
            fun1 = np.log(x).sum() + n * (dc - digamma(a)) 
            fun2 = np.log1p(-x).sum() + n * (dc - digamma(c-a))
            fun3 = np.log(y).sum() + n * (dc - digamma(b))
            fun4 = np.log1p(-y).sum() + n * (dc - digamma(c-b))
            return np.array([fun1, fun2, fun3, fun4])

        alpha4_hat = (rho * np.sqrt(a*b*(c-a)*(c-b)) + (c-a)*(c-b))/c

    def bootstrap_method(self, x, y, B, method, seed=1000, alpha0=None):
        """
        Bootstrap samples for the estimated parameters alpha. It resamples with replacement 
        from x and y B times and for each estimate the parameter. 
        Parameters
        | x (n-array): data in the first component
        | y (n-array): data in the second component
        | B (int): number of bootstrap samples
        | method (fuction): a function that receives arrays x        | and y, and returns an alpha. Pass alpha0 if necessary.
        | seed (int): seed of the random object used in the function.

        Returns
        | boostrap_sample (4xB-array): estimated parameters for each resample.
        """
        ro = np.random.RandomState(seed)
        X = ro.choice(x, size=(len(x), B))
        Y = ro.choice(y, size=(len(y), B))
        bootstrap_sample = np.zeros((4, B))
        for b in range(B):
            if alpha0 is None:
                alpha_hat = method(X[:, b], Y[:, b])
            else:
                alpha_hat = method(X[:, b], Y[:, b], alpha0=alpha0)
            bootstrap_sample[:, b] = alpha_hat
        return bootstrap_sample

def confidence_interval_calculus(level, samples):
    """
    Calculate the percentile interval of level 'level' (for instance level .95).
    Parameters
    | level (float): number between 0 and 1
    | samples (n-array): the array to calculate the percentiles

    Returns
    | ci (2-array): array with the confidemce interval 
    """
    ci = np.quantile(samples, q=[(1-level)/2, (1+level)/2])
    return ci

def experiment_bivbeta(true_alpha, sample_size, monte_carlo_simulations, bootstrap_sample_size, seed):

    bias = np.zeros(4)
    mse = np.zeros(4)
    comp = np.zeros(4)
    coverage = np.zeros(4)

    rng = np.random.default_rng(seed)
    distribution = BivariateBeta()
    for exp in tqdm(range(monte_carlo_simulations)):
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
        alpha =  np.array([alpha_hat1, alpha_hat2, alpha_hat3, alpha_hat4])

        bias_new = true_alpha - alpha
        mse_new = (true_alpha - alpha) * (true_alpha - alpha)
        comp_new = np.array([time1, time2, time3, time4])

        bias = (bias * exp + bias_new)/(exp+1)
        mse = (mse * exp + mse_new)/(mse+1)
        comp = (comp * exp + comp_new)/(exp+1)

        nb = 0
        if exp < nb:

            samples1 = distribution.bootstrap_method(x=X, y=Y, 
                                                     B=bootstrap_sample_size, 
                                                     method=distribution.method_moments_estimator_1, 
                                                     seed=rng.integers(731032178))
            ci1 = confidence_interval_calculus(level=0.95, samples=samples1)
            samples2 = distribution.bootstrap_method(x=X, y=Y, 
                                                     B=bootstrap_sample_size, 
                                                     method=distribution.method_moments_estimator_2, 
                                                     seed=rng.integers(731032178))
            ci2 = confidence_interval_calculus(level=0.95, samples=samples2)
            samples3 = distribution.bootstrap_method(x=X, y=Y, 
                                                     B=bootstrap_sample_size, 
                                                     method=distribution.method_moments_estimator_3, 
                                                     seed=rng.integers(731032178),
                                                     alpha0=(1,1))
            ci3 = confidence_interval_calculus(level=0.95, samples=samples3)
            samples4 = distribution.bootstrap_method(x=X, y=Y, 
                                                     B=bootstrap_sample_size, 
                                                     method=distribution.method_moments_estimator_4, 
                                                     seed=rng.integers(731032178),
                                                     alpha0=(1,1,1,1))
            ci4 = confidence_interval_calculus(level=0.95, samples=samples4)

            if ci1[0] < true_alpha[0] and ci1[1] > true_alpha[0]:
                coverage[0] += 1/nb
            if ci2[0] < true_alpha[1] and ci2[1] > true_alpha[1]:
                coverage[1] += 1/nb
            if ci3[0] < true_alpha[2] and ci3[1] > true_alpha[2]:
                coverage[2] += 1/nb
            if ci4[0] < true_alpha[3] and ci4[1] > true_alpha[3]:
                coverage[3] += 1/nb

    return bias, mse, comp, coverage

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
    monte_carlo_simulations = 10000
    B = 500
    seed = 8392
    bias, mse, comp, coverage = experiment_bivbeta(true_alpha, sample_size, monte_carlo_simulations, B, seed)
    print(bias, mse, comp, coverage)