#usr/bin/env python3
"""Solution to system with m1, m2 and rho

Author: Lucas Moschen

This script is a support for the (unpublished) paper "On a bivariate beta" 
from Lucas Machado Moschen and Luiz Max Carvalho. It plots the figures used 
in the paper.

This script requires that `numpy` and `sympy` be installed within the Python 
environment you are running.

The object is to solve the system 

m1 = (alpha1 + alpha2)/alpha_sum
m2 = (alpha1 + alpha3)/alpha_sum
rho = (alpha1 * alpha4 - alpha2 * alpha3)/(alpha_sum^2 * sqrt(m1(1-m1)m2(1-m2)))
alpha_sum = alpha1 + alpha2 + alpha3 + alpha4
"""

import sympy as sp

if __name__ == '__main__':
    m1, m2, rho, alpha3, alpha4 = sp.symbols('m1 m2 rho alpha3 alpha4')
    alpha1 = (m1 + m2 - 1)/(1 - m1) * alpha3 + m2/(1 - m1) * alpha4
    alpha2 = (1 - m2)/(1 - m1) * alpha3 + (m1 - m2)/(1 - m1) * alpha4
    alpha_sum = sp.simplify(alpha1 + alpha2 + alpha3 + alpha4)
    expression = rho - (alpha1 * alpha4 - alpha2 * alpha3) / (alpha_sum**2 * sp.sqrt(m1 * m2 * (1-m1) * (1 - m2)))
    alpha3 = sp.simplify(sp.solve(expression, alpha3)[0])
    alpha1 = sp.simplify((m1 + m2 - 1)/(1 - m1) * alpha3 + m2/(1 - m1) * alpha4)
    alpha2 = sp.simplify((1 - m2)/(1 - m1) * alpha3 + (m1 - m2)/(1 - m1) * alpha4)
    print('alpha1 = {}'.format(alpha1))
    print('alpha2 = {}'.format(alpha2))
    print('alpha3 = {}'.format(alpha3))
    