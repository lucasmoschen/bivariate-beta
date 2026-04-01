# Energy-GoF Summary

This note summarizes the current energy-based sample-vs-fitted-model goodness-of-fit runs across the configurations:

- `M = 5000`, `B = 999`
- `M = 10000`, `B = 999`
- `M = 10000`, `B = 4999`
- `M = 10000`, `B = 9999`

The p-values below are permutation p-values from the two-sample energy test between:

- sample 1: observed data
- sample 2: one large synthetic sample from the fitted model

These are formal discrepancy summaries for the fitted models. They are not composite-null parametric-bootstrap p-values with refitting.

The main narrative tables below still use `M = 10000`, `B = 999`, but the later sections record all available configurations in full.

## Source Tables

- `experiments/energy_gof/caseA_energy_gof_M10000_B999.csv`
- `experiments/energy_gof/caseA_energy_gof_M10000_B4999.csv`
- `experiments/energy_gof/caseA_energy_gof_M10000_B9999.csv`
- `experiments/energy_gof/caseA_energy_gof_M5000_B999.csv`
- `experiments/energy_gof/caseB_energy_gof_M10000_B999.csv`
- `experiments/energy_gof/caseB_energy_gof_M10000_B4999.csv`
- `experiments/energy_gof/caseB_energy_gof_M10000_B9999.csv`
- `experiments/energy_gof/caseB_energy_gof_M5000_B999.csv`

## Detailed Implementation

The implementation lives in:

- `scripts/python/energy_gof_real_data.py`

The workflow is:

1. load the observed data for Case A or Case B
2. fit each estimator on the observed sample
3. simulate one synthetic reference sample of size `M`
4. compute the ordinary two-sample energy statistic on the pooled Euclidean distances
5. estimate a permutation p-value by repeatedly relabeling the pooled sample into groups of sizes `n` and `M`

The script reports:

- `energy_statistic`
- `perm_null_sd`
- `energy_z_score`
- `p_value`
- `p_value_mcse`

The energy test currently uses the ordinary V-statistic form. The p-value is:

`p = (1 + #{T_perm >= T_obs}) / (B + 1)`

with `B` random permutations.

### Mathematical Formulation

Let the observed sample be

`D_n = {Z_1, ..., Z_n}`, with `Z_i = (x_i, y_i) in (0,1)^2`.

For each estimator or fitted model `e`:

1. fit the parameter on the observed data

   `theta_hat_e = fit_e(D_n)`

2. generate one synthetic reference sample from the fitted model

   `R_M^(e) = {Y_1, ..., Y_M}`, with `Y_j ~ P_(theta_hat_e)`

3. compare `D_n` and `R_M^(e)` using the two-sample energy statistic.

The implemented energy statistic is the ordinary V-statistic:

`T_(n,M)(D_n, R_M) = (2/(nM)) sum_(i=1)^n sum_(j=1)^M ||Z_i - Y_j||`

`- (1/n^2) sum_(i=1)^n sum_(i'=1)^n ||Z_i - Z_(i')||`

`- (1/M^2) sum_(j=1)^M sum_(j'=1)^M ||Y_j - Y_(j')||`

where `||.||` is the Euclidean norm in `R^2`.

Large values of `T_(n,M)` indicate greater discrepancy between the observed sample and the fitted-model reference sample.

### Permutation Calibration

For each fitted model row:

- form the pooled sample `P = D_n union R_M`
- repeatedly relabel the pooled points into two groups of sizes `n` and `M`
- recompute the energy statistic under each relabeling

If `T_obs` is the observed statistic and `T_1, ..., T_B` are the permutation statistics, then the reported p-value is

`p_hat = (1 + sum_{b=1}^B 1{T_b >= T_obs}) / (B + 1)`.

This is a one-sided permutation p-value because larger energy statistics correspond to worse fit.

The output also reports:

- `perm_null_sd = sd(T_1, ..., T_B)`
- `energy_z_score = (T_obs - mean(T_1, ..., T_B)) / perm_null_sd`
- `p_value_mcse = sqrt(p_hat (1 - p_hat) / (B + 1))`

`p_value_mcse` is the Monte Carlo standard error of the permutation p-value estimator.

### Interpretation Of The Procedure

This is a sample-vs-fitted-model discrepancy analysis. It is not the stricter composite-null procedure in which one would:

- simulate data under the fitted model
- refit the model on every simulated dataset
- recalibrate the test statistic under that refitted null distribution

So the p-values here are useful formal summaries, but they should not be described as exact composite-null GoF p-values that account for estimation uncertainty.

### Data Preparation

Case A:

- source: 2024 DTP3 and measles one-year-old coverage tables
- merge by `Entity`, `Code`, `Year`
- divide percentages by `100`
- if `x == y`, perturb `x` by `U(-10^{-6}, 10^{-6})`

Case B:

- source: ANVISA COVID-19 diagnostic tables
- parse sensitivity and specificity point estimates from the CI strings
- if sensitivity or specificity is exactly `1`, subtract `U(10^{-6}, 5 \\times 10^{-4})`

### Estimator Definitions Used Here

- `beta_bayes_mean`: Bayesian bivariate-beta plug-in estimator using `bivariate-beta-model-lognormal.stan`, with posterior mean of `alpha`
- `beta_bayes_median`: same posterior fit, using componentwise posterior median of `alpha`
- `bln_bayes`: Bayesian bivariate logit-normal plug-in estimator from `logit-bivariate-model.stan`, using posterior mean of `(mu_1, mu_2, sigma_1, sigma_2, rho)`
- `mom_closed`: `BivariateBeta.method_moments_estimator_1`
- `mom_nls`: `BivariateBeta.method_moments_estimator_4`
- `mle`: closed-form logit-normal MLE on the logit scale
- `ind_beta_mle`: independent Beta marginal MLE benchmark used in the notebook

Important implementation notes:

- `beta_bayes_mean` and `beta_bayes_median` now come from the same Stan fit for a given case, then take mean vs median from the same posterior draws.
- `mle` BLN is now computed in closed form rather than via Stan optimization.
- when a fitted bivariate-beta parameter has an exact zero component, the stored estimate is unchanged, but the synthetic Dirichlet simulation uses a tiny positive floor `10^{-8}` only so simulation remains defined.

### Mathematical Description Of Each Estimator

`beta_bayes_mean` and `beta_bayes_median`:

- The Bayesian bivariate-beta model is parameterized by `alpha = s theta`, where `theta` lies on the simplex and `s > 0`.
- The prior is `theta ~ Dirichlet(1,1,1,1)` and `s ~ lognormal(log(4), 1)`.
- The plug-in estimators are:
  - `alpha_hat_mean = posterior mean of alpha`
  - `alpha_hat_median = componentwise posterior median of alpha`

`bln_bayes`:

- On the logit scale, `W_i = (logit(x_i), logit(y_i))` is modeled as bivariate normal with parameters `(mu, Sigma)`.
- The plug-in estimator used here is the posterior mean of `(mu_1, mu_2, sigma_1, sigma_2, rho)`.

`mle` BLN:

- The logistic-normal likelihood differs from the Gaussian likelihood on the logit scale only by a Jacobian term that does not depend on the model parameters.
- Therefore the MLE is the Gaussian MLE on the transformed sample `W_i = (logit(x_i), logit(y_i))`.
- The closed-form estimates are:

  `mu_hat = (1/n) sum_i W_i`

  `Sigma_hat = (1/n) sum_i (W_i - mu_hat)(W_i - mu_hat)^T`

  with

  `sigma_1 = sqrt(Sigma_hat,11)`, `sigma_2 = sqrt(Sigma_hat,22)`, `rho = Sigma_hat,12 / (sigma_1 sigma_2)`.

`mom_closed` and `mom_nls`:

- These are exactly the estimators implemented in `parameter_estimation.py`.
- `mom_closed` uses `method_moments_estimator_1`.
- `mom_nls` uses `method_moments_estimator_4`.

`ind_beta_mle`:

- Fit a univariate Beta distribution to each margin independently by marginal MLE.
- Set the joint model to the product of the two fitted marginals, so the implied correlation is zero.

## Detailed Chosen Parameters

### Main Narrative Tables Summarized Here

- `M = 10000`
- `B = 999`
- energy variant: V-statistic
- base seed: `20260331`

### Bayesian Bivariate-Beta Settings

- Stan model: `scripts/stan/bivariate-beta-model-lognormal.stan`
- prior on `theta`: `Dirichlet(1,1,1,1)`
- prior on `s`: `lognormal(log(4), 1)`
- chains: `4`
- warmup: `2000`
- sampling: `2000`
- `adapt_delta = 0.95` for Case A
- `adapt_delta = 0.99` for Case B

### Bayesian BLN Settings

- Stan model: `scripts/stan/logit-bivariate-model.stan`
- chains: `4`
- warmup: `1000`
- sampling: `2000`
- priors enabled
- plug-in summary used here: posterior mean

### Case-Specific Preprocessing Parameters

- Case A year: `2024`
- Case A tie jitter seed: `13`
- Case A tie jitter amplitude: `10^{-6}`
- Case B one-boundary jitter seed: `13`
- Case B upper jitter bound: `5 \\times 10^{-4}`

### Exact Command

```bash
python scripts/python/energy_gof_real_data.py --case all --reference-size 10000 --permutations 999
```

## Observed Sample Moments

### Case A Observed Moments

| m1 | m2 | v1 | v2 | rho |
|---:|---:|---:|---:|---:|
| 0.865487 | 0.858872 | 0.016856 | 0.019966 | 0.854930 |

### Case B Observed Moments

| m1 | m2 | v1 | v2 | rho |
|---:|---:|---:|---:|---:|
| 0.946821 | 0.942876 | 0.004221 | 0.017282 | -0.186862 |

## Case A

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.002076 | 0.000741 | 1.331556 | 0.094 | 0.009228 |
| beta_bayes_median | 10000 | 0.001823 | 0.000762 | 1.000766 | 0.121 | 0.010313 |
| bln_bayes | 10000 | 0.000660 | 0.000836 | -0.560471 | 0.695 | 0.014559 |
| mom_closed | 10000 | 0.000898 | 0.000805 | -0.190125 | 0.427 | 0.015642 |
| mom_nls | 10000 | 0.000754 | 0.000784 | -0.423103 | 0.547 | 0.015741 |
| mle | 10000 | 0.000664 | 0.000805 | -0.621583 | 0.723 | 0.014152 |

### Case A findings

- The smallest energy discrepancies are for `bln_bayes`, `mle`, and `mom_nls`.
- `mom_closed` is very close to that group in absolute terms.
- The difference between `bln_bayes`, `mle`, `mom_nls`, and `mom_closed` is small on the raw energy scale.
- `beta_bayes_mean` and `beta_bayes_median` are worse in the current plug-in comparison, but this should be interpreted cautiously: these rows are based on point summaries of the posterior (`alpha` mean or componentwise median), not the full posterior predictive distribution.
- Visual diagnostics still matter for Case A. The energy ranking does not by itself rule out concerns about BLN oversmoothing into sparse regions.

### Case A Fitted Parameters

| model | fitted parameters |
|---|---|
| beta_bayes_mean | `alpha = (5.4395, 0.2065, 0.1762, 0.7513)` |
| beta_bayes_median | `alpha = (5.4266, 0.2058, 0.1756, 0.7497)` |
| bln_bayes | `(mu1, mu2, sigma1, sigma2, rho) = (2.3634, 2.3304, 1.2604, 1.3124, 0.8161)` |
| mom_closed | `alpha = (4.9907, 0.1216, 0.0825, 0.7120)` |
| mom_nls | `alpha = (4.6002, 0.1128, 0.0754, 0.6563)` |
| mle | `(mu1, mu2, sigma1, sigma2, rho) = (2.3624, 2.3304, 1.2523, 1.3036, 0.8203)` |

### Case A Model-Implied Moments

| model | fit_m1 | fit_m2 | fit_v1 | fit_v2 | fit_rho |
|---|---:|---:|---:|---:|---:|
| beta_bayes_mean | 0.858908 | 0.854295 | 0.016001 | 0.016435 | 0.763150 |
| beta_bayes_median | 0.858903 | 0.854296 | 0.016035 | 0.016470 | 0.763454 |
| bln_bayes | 0.864367 | 0.857807 | 0.019395 | 0.021834 | 0.783308 |
| mom_closed | 0.865487 | 0.858872 | 0.016856 | 0.017549 | 0.854930 |
| mom_nls | 0.865623 | 0.858744 | 0.018049 | 0.018822 | 0.854930 |
| mle | 0.862778 | 0.855809 | 0.019408 | 0.022004 | 0.791126 |

## Case B

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.006454 | 0.001329 | 3.257540 | 0.016 | 0.003968 |
| beta_bayes_median | 10000 | 0.006401 | 0.001300 | 3.294062 | 0.014 | 0.003715 |
| ind_beta_mle | 10000 | 0.005052 | 0.001185 | 2.623885 | 0.024 | 0.004840 |
| bln_bayes | 10000 | 0.010460 | 0.002023 | 3.622501 | 0.011 | 0.003298 |
| mom_closed | 10000 | 0.006902 | 0.000872 | 6.179937 | 0.001 | 0.000999 |
| mom_nls | 10000 | 0.036178 | 0.001478 | 22.795652 | 0.001 | 0.000999 |
| mle | 10000 | 0.010705 | 0.002363 | 3.133104 | 0.020 | 0.004427 |

### Case B findings

- `ind_beta_mle` has the smallest energy discrepancy among the current rows.
- `beta_bayes_mean` and `beta_bayes_median` are close to each other and improve substantially over the BLN rows.
- `bln_bayes` and `mle` are clearly worse than the independent-Beta benchmark on this energy metric.
- `mom_closed` is distinctly worse than the Bayesian BB rows.
- `mom_nls` is a clear failure on Case B under this discrepancy measure.

### Case B Fitted Parameters

| model | fitted parameters |
|---|---|
| beta_bayes_mean | `alpha = (4.3035, 0.1621, 0.2285, 0.1472)` |
| beta_bayes_median | `alpha = (4.2586, 0.1603, 0.2267, 0.1460)` |
| ind_beta_mle | `(a1, b1, a2, b2) = (7.1335, 0.3918, 4.4257, 0.3026)` |
| bln_bayes | `(mu1, mu2, sigma1, sigma2, rho) = (4.4835, 4.8206, 2.8718, 2.7332, 0.0750)` |
| mom_closed | `alpha = (9.7223, 0.6242, 0.5811, 0.0000)` |
| mom_nls | `alpha = (5.8846, 0.7598, 0.7327, 0.0000)` |
| mle | `(mu1, mu2, sigma1, sigma2, rho) = (4.5149, 4.8411, 2.8256, 2.6872, 0.0796)` |

### Case B Model-Implied Moments

| model | fit_m1 | fit_m2 | fit_v1 | fit_v2 | fit_rho |
|---|---:|---:|---:|---:|---:|
| beta_bayes_mean | 0.922400 | 0.936108 | 0.012254 | 0.010239 | 0.388926 |
| beta_bayes_median | 0.922215 | 0.936072 | 0.012386 | 0.010332 | 0.389235 |
| ind_beta_mle | 0.947931 | 0.936001 | 0.005790 | 0.010457 | 0.000000 |
| bln_bayes | 0.907840 | 0.931385 | 0.034272 | 0.023393 | 0.049264 |
| mom_closed | 0.946821 | 0.942876 | 0.004221 | 0.004516 | -0.058333 |
| mom_nls | 0.900683 | 0.897010 | 0.010678 | 0.011028 | -0.112519 |
| mle | 0.909605 | 0.931671 | 0.034145 | 0.023194 | 0.059680 |

## Stability When Increasing M From 5000 To 10000

The main `M = 5000` to `M = 10000` comparison shows:

- For Case A, the raw energy statistics for `mom_closed`, `mom_nls`, and `mle` barely changed.
- For Case B, the raw energy statistics changed only modestly for most rows and the qualitative ranking stayed the same.
- The p-values move more than the raw energy statistics, which is expected because the p-value depends on the permutation calibration as well as the observed statistic.

This suggests that moving from `M = 5000` to `M = 10000` already captured most of the gain available from enlarging the synthetic reference sample.

Increasing `B` from `999` to `4999` refines the p-values and their Monte Carlo error, but it does not change the observed energy statistic itself.

## Precision Recommendation

If higher precision is needed, the first thing to increase is the permutation count, not `M`.

Recommended next run:

- keep `M = 10000`
- increase `B` from `999` to `4999` or `9999`

Reason:

- `M` mainly controls Monte Carlo noise in the synthetic reference sample
- `B` mainly controls Monte Carlo noise in the p-value estimate
- with `B = 999`, the reported `p_value_mcse` is still around `0.01` to `0.016` for the non-extreme Case A p-values

Practical advice:

- If the goal is a sharper p-value, increase permutations.
- If the goal is to check stability of the statistic itself, a second run with the same `M = 10000` and a different seed is likely more informative than pushing `M` much larger.
- Increasing `M` much beyond `10000` is not the first recommendation here because the current implementation uses the full pooled distance matrix, so memory and runtime can grow quickly.

## Suggested Final Rerun

For a higher-precision final table:

```bash
python scripts/python/energy_gof_real_data.py --case all --reference-size 10000 --permutations 4999
```

If runtime is still acceptable:

```bash
python scripts/python/energy_gof_real_data.py --case all --reference-size 10000 --permutations 9999
```

## Interpretation Guidance

- Rank models primarily by `energy_statistic`.
- Use `energy_z_score` and `perm_null_sd` as scale diagnostics.
- Use `p_value` as a formal discrepancy summary, but do not over-interpret small differences in p-value when the raw energy statistics are very close.

## Complete Result Tables By Configuration

This section records the main GoF output tables for all currently used configurations.

### Case A, `M = 5000`, `B = 999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 5000 | 0.002446 | 0.000747 | 1.814979 | 0.0540 | 0.007147 |
| beta_bayes_median | 5000 | 0.001910 | 0.000741 | 1.132228 | 0.1030 | 0.009612 |
| bln_bayes | 5000 | 0.000587 | 0.000794 | -0.681207 | 0.7900 | 0.012880 |
| mom_closed | 5000 | 0.000894 | 0.000767 | -0.207337 | 0.4450 | 0.015715 |
| mom_nls | 5000 | 0.000769 | 0.000831 | -0.381974 | 0.5510 | 0.015729 |
| mle | 5000 | 0.000655 | 0.000887 | -0.593735 | 0.7080 | 0.014378 |

### Case A, `M = 10000`, `B = 999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.002076 | 0.000741 | 1.331556 | 0.0940 | 0.009228 |
| beta_bayes_median | 10000 | 0.001823 | 0.000762 | 1.000766 | 0.1210 | 0.010313 |
| bln_bayes | 10000 | 0.000660 | 0.000836 | -0.560471 | 0.6950 | 0.014559 |
| mom_closed | 10000 | 0.000898 | 0.000805 | -0.190125 | 0.4270 | 0.015642 |
| mom_nls | 10000 | 0.000754 | 0.000784 | -0.423103 | 0.5470 | 0.015741 |
| mle | 10000 | 0.000664 | 0.000805 | -0.621583 | 0.7230 | 0.014152 |

### Case A, `M = 10000`, `B = 4999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.002076 | 0.000761 | 1.303722 | 0.0910 | 0.004067 |
| beta_bayes_median | 10000 | 0.001823 | 0.000747 | 1.032002 | 0.1168 | 0.004542 |
| bln_bayes | 10000 | 0.000660 | 0.000782 | -0.582094 | 0.6956 | 0.006508 |
| mom_closed | 10000 | 0.000898 | 0.000800 | -0.194928 | 0.4366 | 0.007014 |
| mom_nls | 10000 | 0.000754 | 0.000811 | -0.394782 | 0.5582 | 0.007023 |
| mle | 10000 | 0.000664 | 0.000797 | -0.574679 | 0.6970 | 0.006499 |

### Case A, `M = 10000`, `B = 9999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.002076 | 0.000749 | 1.338890 | 0.0887 | 0.002843 |
| beta_bayes_median | 10000 | 0.001823 | 0.000752 | 1.017203 | 0.1190 | 0.003238 |
| bln_bayes | 10000 | 0.000660 | 0.000792 | -0.584657 | 0.6967 | 0.004597 |
| mom_closed | 10000 | 0.000898 | 0.000784 | -0.191036 | 0.4339 | 0.004956 |
| mom_nls | 10000 | 0.000754 | 0.000810 | -0.396316 | 0.5579 | 0.004966 |
| mle | 10000 | 0.000664 | 0.000802 | -0.571499 | 0.6945 | 0.004606 |

### Case B, `M = 5000`, `B = 999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 5000 | 0.006622 | 0.001506 | 2.902621 | 0.0210 | 0.004534 |
| beta_bayes_median | 5000 | 0.006394 | 0.001414 | 2.975790 | 0.0190 | 0.004317 |
| ind_beta_mle | 5000 | 0.005467 | 0.001139 | 3.141343 | 0.0200 | 0.004427 |
| bln_bayes | 5000 | 0.009789 | 0.001990 | 3.398800 | 0.0150 | 0.003844 |
| mom_closed | 5000 | 0.007067 | 0.000854 | 6.476859 | 0.0010 | 0.000999 |
| mom_nls | 5000 | 0.035957 | 0.001441 | 23.254911 | 0.0010 | 0.000999 |
| mle | 5000 | 0.010957 | 0.002142 | 3.616005 | 0.0140 | 0.003715 |

### Case B, `M = 10000`, `B = 999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.006454 | 0.001329 | 3.257540 | 0.0160 | 0.003968 |
| beta_bayes_median | 10000 | 0.006401 | 0.001300 | 3.294062 | 0.0140 | 0.003715 |
| ind_beta_mle | 10000 | 0.005052 | 0.001185 | 2.623885 | 0.0240 | 0.004840 |
| bln_bayes | 10000 | 0.010460 | 0.002023 | 3.622501 | 0.0110 | 0.003298 |
| mom_closed | 10000 | 0.006902 | 0.000872 | 6.179937 | 0.0010 | 0.000999 |
| mom_nls | 10000 | 0.036178 | 0.001478 | 22.795652 | 0.0010 | 0.000999 |
| mle | 10000 | 0.010705 | 0.002363 | 3.133104 | 0.0200 | 0.004427 |

### Case B, `M = 10000`, `B = 4999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.006454 | 0.001399 | 3.066878 | 0.0176 | 0.001860 |
| beta_bayes_median | 10000 | 0.006401 | 0.001365 | 3.108565 | 0.0190 | 0.001931 |
| ind_beta_mle | 10000 | 0.005052 | 0.001220 | 2.532720 | 0.0282 | 0.002341 |
| bln_bayes | 10000 | 0.010460 | 0.002156 | 3.395648 | 0.0122 | 0.001552 |
| mom_closed | 10000 | 0.006902 | 0.000884 | 6.094169 | 0.0016 | 0.000565 |
| mom_nls | 10000 | 0.036178 | 0.001441 | 23.392042 | 0.0002 | 0.000200 |
| mle | 10000 | 0.010705 | 0.002257 | 3.311410 | 0.0152 | 0.001730 |

### Case B, `M = 10000`, `B = 9999`

| model | M | energy_statistic | perm_null_sd | energy_z_score | p_value | p_value_mcse |
|---|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 10000 | 0.006454 | 0.001379 | 3.108213 | 0.0177 | 0.001319 |
| beta_bayes_median | 10000 | 0.006401 | 0.001381 | 3.071920 | 0.0183 | 0.001340 |
| ind_beta_mle | 10000 | 0.005052 | 0.001218 | 2.547843 | 0.0285 | 0.001664 |
| bln_bayes | 10000 | 0.010460 | 0.002138 | 3.421697 | 0.0116 | 0.001071 |
| mom_closed | 10000 | 0.006902 | 0.000888 | 6.070701 | 0.0015 | 0.000387 |
| mom_nls | 10000 | 0.036178 | 0.001434 | 23.528481 | 0.0001 | 0.000100 |
| mle | 10000 | 0.010705 | 0.002231 | 3.363278 | 0.0141 | 0.001179 |

## Cross-Configuration Comparison Tables

These tables show how the reported statistic and p-value move when `M` and `B` are changed.

### Case A Cross-Configuration Comparison

| model | stat_M5000_B999 | p_M5000_B999 | stat_M10000_B999 | p_M10000_B999 | stat_M10000_B4999 | p_M10000_B4999 | stat_M10000_B9999 | p_M10000_B9999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 0.002446 | 0.0540 | 0.002076 | 0.0940 | 0.002076 | 0.0910 | 0.002076 | 0.0887 |
| beta_bayes_median | 0.001910 | 0.1030 | 0.001823 | 0.1210 | 0.001823 | 0.1168 | 0.001823 | 0.1190 |
| bln_bayes | 0.000587 | 0.7900 | 0.000660 | 0.6950 | 0.000660 | 0.6956 | 0.000660 | 0.6967 |
| mom_closed | 0.000894 | 0.4450 | 0.000898 | 0.4270 | 0.000898 | 0.4366 | 0.000898 | 0.4339 |
| mom_nls | 0.000769 | 0.5510 | 0.000754 | 0.5470 | 0.000754 | 0.5582 | 0.000754 | 0.5579 |
| mle | 0.000655 | 0.7080 | 0.000664 | 0.7230 | 0.000664 | 0.6970 | 0.000664 | 0.6945 |

### Case B Cross-Configuration Comparison

| model | stat_M5000_B999 | p_M5000_B999 | stat_M10000_B999 | p_M10000_B999 | stat_M10000_B4999 | p_M10000_B4999 | stat_M10000_B9999 | p_M10000_B9999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| beta_bayes_mean | 0.006622 | 0.0210 | 0.006454 | 0.0160 | 0.006454 | 0.0176 | 0.006454 | 0.0177 |
| beta_bayes_median | 0.006394 | 0.0190 | 0.006401 | 0.0140 | 0.006401 | 0.0190 | 0.006401 | 0.0183 |
| ind_beta_mle | 0.005467 | 0.0200 | 0.005052 | 0.0240 | 0.005052 | 0.0282 | 0.005052 | 0.0285 |
| bln_bayes | 0.009789 | 0.0150 | 0.010460 | 0.0110 | 0.010460 | 0.0122 | 0.010460 | 0.0116 |
| mom_closed | 0.007067 | 0.0010 | 0.006902 | 0.0010 | 0.006902 | 0.0016 | 0.006902 | 0.0015 |
| mom_nls | 0.035957 | 0.0010 | 0.036178 | 0.0010 | 0.036178 | 0.0002 | 0.036178 | 0.0001 |
| mle | 0.010957 | 0.0140 | 0.010705 | 0.0200 | 0.010705 | 0.0152 | 0.010705 | 0.0141 |
