# Bivariate beta: parameter inference and diagnostics

By Lucas Machado Moschen and Luiz Max Carvalho. 

Correlated proportions appear in many real-world applications and present a unique challenge in terms of finding an appropriate probabilistic model due to their constrained nature. 
The bivariate beta is a natural extension of the well-known beta distribution to the space of correlated quantities on $[0,1]^2$. 
Its construction is not unique, however. 
Over the years, many bivariate beta distributions have been proposed, ranging from three to eight or more parameters, and for which the joint density and distribution moments vary in terms of mathematical tractability. 
In this paper, we investigate the construction proposed by Olkin & Trikalinos (2015), which strikes a balance between parameter-richness and tractability.
We provide classical (frequentist) and Bayesian approaches to estimation in the form of method-of-moments and latent variable/data augmentation coupled with Hamiltonian Monte Carlo, respectively. 
The elicitation of bivariate beta as a prior distribution is also discussed. 
The development of diagnostics for checking model fit and adequacy is explored in depth with the aid of Monte Carlo experiments under both well-specified and misspecified data-generating settings.

This repository contains all the codes, experiments and images used in the paper. For more details on the problem, see [the pdf in the arXiv](https://arxiv.org/pdf/2303.01271.pdf).

## Structure 

The main pieces of the repository as follows: 

```{bash}
├───data
├───experiments
├───figures
├───notebooks
    |───bayesian_estimation_biv_beta.ipynb
    |───images_paper.ipynb
    │───prior-predictive-checking.ipynb
    │───usage.ipynb
└───scripts
    ├───python
    |   |───__init__.py
    │   |───experiments.py
    │   |───parameter_estimation.py
    ├───r
    └───stan
```

This structure includes:

* `data`: bivariate data from sensitivity and specificity for trying out the `BivariateBeta` class;
* `experiments`: all the experiments in the paper, which resulted in the tables and some insights are organized in this sub-folder;
* `notebooks`: the bayesian experiments, images and data examples:
   1. `bayesian_estimation_biv_beta.ipynb`: Simulation based calibration and other experiments with the Bayesian implementation in Stan. 
   2. `images_paper.ipynb`: Generate all the images that are on the paper.
   3. `prior-predictive-checking.ipynb`: Prior analysis for the parameter of the distribution.
   4. `usage.ipynb`: File exemplifying how to use the class `BivariateBeta`.
* `scripts`: scripts written in Python, R and Stan. The more important are:
   1. `experiments.py`: all the experiments contained in the paper are generated using this code.;
   2. `parameter_estimation.py`: defines the class `BivariateBeta` with all the necessary material for parameter estimation.

## Hands on

We expect it is the most reproducible that it can the be. To experiment, one needs: 

- Python 3 and Anaconda are essentials. 
- Clone this repository in your machine with the command ```git clone https://github.com/lucasmoschen/bivariate-beta``` 
- Create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from the file `environment.yml`. Install the requirements with ```conda env create -f environment.yml```
- Installing `cmdstan` and `cmdstanpy` can be tricky, so check [here](https://mc-stan.org/cmdstanpy/).

## Experimenting 

For a quick try out of the codes, considering using the notebook `usage.ipynb` which demonstrates the main aspects of the work. 
The file `experiments.py` systematizes the experiments. 
For instance, go to the `if __name__ === "__main__"` part and changes the following code according to your specifications:

```{python}
monte_carlo_size = 1000
bootstrap_size = 500
seed = 378291

true_alpha = np.array([1,1,1,1])
experiment_bivbeta(true_alpha, 50, monte_carlo_size, bootstrap_size, seed)
experiment_bivbeta(true_alpha, 200, monte_carlo_size, bootstrap_size, seed)
```

After that, run the code `python experiments.py` and the results will be placed on `experiments` folder as a JSON file.

## Suggestions

Please, for any suggestions, write an issue, and I will answer as quickly as I can. Thanks!