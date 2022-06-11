#usr/bin/env python3

from importlib.metadata import distribution
import numpy as np
import matplotlib.pyplot as plt
from parameter_estimation import BivariateBeta

if __name__ == '__main__':

    true_alpha = np.array([1,1,1,1])
    n=1000
    U = np.random.dirichlet(true_alpha, size=n)
    X = U[:,0] + U[:,1]
    Y = U[:,0] + U[:,2]
    m1 = X.mean()
    m2 = Y.mean()
    v1 = X.var(ddof=1)
    v2 = Y.var(ddof=1)
    rho = np.corrcoef(X,Y)[0,1]

    distribution = BivariateBeta()
    alpha1, alpha2 = np.meshgrid(np.linspace(1e-1, 2, 100), np.linspace(1e-1, 2, 100))
    v = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            true_alpha = np.array([alpha1[i,j], alpha2[i,j], 1, 1])
            v[i,j] = distribution.loss_function(alpha=true_alpha, m1=m1, m2=m2, v1=v1, v2=v2, rho=rho, g=distribution._choose_loss_function())
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(alpha1, alpha2, v)
    fig.colorbar(cp)
    plt.show()
