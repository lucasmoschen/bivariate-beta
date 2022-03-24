#usr/bin/env python3

from tarfile import XGLTYPE
from telnetlib import XDISPLOC
import numpy as np
from parameter_estimation import BivariateBeta
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    
    true_alpha = np.array([2,4,3,1])
    ro = np.random.default_rng(np.random.randint(31829))
    distribution = BivariateBeta()
    sample_sizes = [10, 20, 50, 100, 200, 1000]
    bootstrap_samples = list(range(50, 5000, 50))
    ci_alpha1 = np.zeros((2, len(bootstrap_samples)))

    fig, ax = plt.subplots(2, 3)

    for ind, sample_size in tqdm(enumerate(sample_sizes)):
        i = ind // 3
        j = ind % 3
        U = ro.dirichlet(true_alpha, size=sample_size)
        X = U[:, 0] + U[:, 1]
        Y = U[:, 0] + U[:, 2]
        for b in bootstrap_samples:
            samples = distribution.bootstrap_method(X, Y, B=b, method=distribution.method_moments_estimator_1, processes=4)
            ci_alpha1[:, (b-50)//50] = distribution.confidence_interval(level=0.95, samples=samples)[:, 0]
        ax[i,j].plot(bootstrap_samples, ci_alpha1[0])
        ax[i,j].plot(bootstrap_samples, ci_alpha1[1])
        ax[i,j].set_title('n={}'.format(sample_size))
    plt.show()

if __name__ == '__main__':
    main()