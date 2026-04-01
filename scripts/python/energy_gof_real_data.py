#!/usr/bin/env python3
"""Formal energy-based goodness-of-fit checks for the real-data case studies.

This script implements the workflow:

1. fit each model/estimator on the observed sample;
2. simulate one large synthetic reference sample from the fitted model;
3. compare observed and synthetic samples with a two-sample energy test;
4. report the fitted parameter, energy statistic, p-value, and reference size.

The resulting p-values are sample-vs-fitted-model discrepancy summaries. They
are not composite-null parametric-bootstrap p-values with refitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
try:
    from cmdstanpy import CmdStanModel
except ImportError:
    CmdStanModel = None
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import betaln, expit, logit
from scipy.stats import beta as beta_dist
from tqdm.auto import trange

from parameter_estimation import BivariateBeta


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
STAN_DIR = REPO_ROOT / "scripts" / "stan"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "energy_gof"

CASE_A_DEFAULT_MODELS = [
    "beta_bayes_mean",
    "beta_bayes_median",
    "bln_bayes",
    "mom_closed",
    "mom_nls",
    "mle",
]

CASE_B_DEFAULT_MODELS = [
    "beta_bayes_mean",
    "beta_bayes_median",
    "ind_beta_mle",
    "bln_bayes",
    "mom_closed",
    "mom_nls",
    "mle",
]

CASE_B_EPS_MAX = 5e-4
CASE_B_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
DIRICHLET_ALPHA_FLOOR = 1e-8
LOGIT_NORMAL_SCALE_FLOOR = 1e-10
LOGIT_NORMAL_RHO_CLIP = 1.0 - 1e-12

MODEL_SEED_OFFSET = {
    "beta_bayes": 101,
    "beta_bayes_mean": 103,
    "beta_bayes_median": 107,
    "bln_bayes": 211,
    "mom_closed": 307,
    "mom_nls": 401,
    "mle": 503,
    "ind_beta_mle": 601,
}

CASE_SEED_OFFSET = {
    "caseA": 10_000,
    "caseB": 20_000,
}

MODEL_LABEL = {
    "beta_bayes": "Bayesian BB",
    "beta_bayes_mean": "Bayesian BB Mean",
    "beta_bayes_median": "Bayesian BB Median",
    "bln_bayes": "Bayesian BLN",
    "mom_closed": "MoM-Closed BB",
    "mom_nls": "MoM-NLS BB",
    "mle": "MLE BLN",
    "ind_beta_mle": "Independent Beta",
}


@dataclass
class Dataset:
    case_id: str
    case_label: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y]).astype(float)

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


@dataclass
class ModelResult:
    model_key: str
    model_label: str
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    reference_size: int
    reference_sample: np.ndarray
    fit_metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["caseA", "caseB", "all"],
        default="all",
        help="Which real-data case study to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_LABEL),
        help="Optional explicit model list. If omitted, case-specific defaults are used.",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=5000,
        help="Synthetic reference sample size M used for every model.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Number of random permutations used for the energy-test p-value.",
    )
    parser.add_argument(
        "--energy-variant",
        choices=["v", "u"],
        default="v",
        help="Use the ordinary V-statistic or the unbiased U-statistic variant.",
    )
    parser.add_argument(
        "--bayes-point-estimate",
        choices=["mean", "median"],
        default="mean",
        help="Point estimate extracted from posterior draws for `bln_bayes` and the legacy `beta_bayes` alias.",
    )
    parser.add_argument(
        "--bb-draws",
        type=int,
        default=2000,
        help="Posterior draws per chain for the Bayesian bivariate-beta fit.",
    )
    parser.add_argument(
        "--bb-warmup",
        type=int,
        default=2000,
        help="Warmup draws per chain for the Bayesian bivariate-beta fit.",
    )
    parser.add_argument(
        "--bb-adapt-delta-case-a",
        type=float,
        default=0.95,
        help="Stan adapt_delta for the Case A Bayesian bivariate-beta fit.",
    )
    parser.add_argument(
        "--bb-adapt-delta-case-b",
        type=float,
        default=0.99,
        help="Stan adapt_delta for the Case B Bayesian bivariate-beta fit.",
    )
    parser.add_argument(
        "--bln-draws",
        type=int,
        default=2000,
        help="Posterior draws per chain for the Bayesian BLN fit.",
    )
    parser.add_argument(
        "--bln-warmup",
        type=int,
        default=1000,
        help="Warmup draws per chain for the Bayesian BLN fit.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of Stan chains for Bayesian fits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260331,
        help="Base seed for fitting, simulation, and permutation testing.",
    )
    parser.add_argument(
        "--case-a-year",
        type=int,
        default=2024,
        help="Year used for the Case A immunization data.",
    )
    parser.add_argument(
        "--case-a-seed",
        type=int,
        default=13,
        help="Seed used for the Case A tie-breaking jitter.",
    )
    parser.add_argument(
        "--case-a-tie-epsilon",
        type=float,
        default=1e-6,
        help="Tie-breaking perturbation magnitude used when x == y in Case A.",
    )
    parser.add_argument(
        "--case-b-seed",
        type=int,
        default=13,
        help="Seed used for the Case B boundary jitter.",
    )
    parser.add_argument(
        "--case-b-eps-max",
        type=float,
        default=CASE_B_EPS_MAX,
        help="Upper bound for the Case B downward jitter when sensitivity/specificity equals 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where case-wise CSV and JSON outputs are written.",
    )
    return parser.parse_args()


def posterior_point_estimate(draws: np.ndarray, how: str) -> np.ndarray:
    draws = np.asarray(draws, dtype=float)
    if how == "mean":
        return draws.mean(axis=0)
    if how == "median":
        return np.median(draws, axis=0)
    raise ValueError(f"Unknown point estimate: {how}")


def load_case_a(year: int, tie_epsilon: float, seed: int) -> Dataset:
    df_dtp = pd.read_csv(
        DATA_DIR / "share-of-children-immunized-dtp3" / "share-of-children-immunized-dtp3.csv",
        names=["Entity", "Code", "Year", "Share"],
        header=0,
    )
    df_measles = pd.read_csv(
        DATA_DIR / "share-of-children-vaccinated-against-measles" / "share-of-children-vaccinated-against-measles.csv",
        names=["Entity", "Code", "Year", "Share"],
        header=0,
    )
    df = pd.merge(df_dtp, df_measles, on=["Entity", "Code", "Year"]).dropna()
    df = df.loc[df["Year"] == year].copy()
    df["Share_x"] = df["Share_x"] / 100.0
    df["Share_y"] = df["Share_y"] / 100.0

    x = df["Share_x"].to_numpy(dtype=float).copy()
    y = df["Share_y"].to_numpy(dtype=float).copy()
    rng = np.random.default_rng(seed)
    ties = x == y
    if np.any(ties):
        x[ties] += rng.uniform(-tie_epsilon, tie_epsilon, size=int(ties.sum()))
    x = np.clip(x, 1e-12, 1 - 1e-12)
    y = np.clip(y, 1e-12, 1 - 1e-12)

    return Dataset(
        case_id="caseA",
        case_label="Case A: share of immunized children",
        x=x,
        y=y,
        metadata={
            "year": int(year),
            "tie_epsilon": float(tie_epsilon),
            "jitter_seed": int(seed),
            "n_ties_jittered": int(ties.sum()),
        },
    )


def _extract_case_b_number(index: int) -> Callable[[str], float]:
    def extractor(text: str) -> float:
        values = CASE_B_NUMBER_PATTERN.findall(str(text).replace(",", "."))
        return float(values[index])

    return extractor


def load_case_b(seed: int, eps_max: float) -> Dataset:
    folder = DATA_DIR / "covid_tests_data" / "raw"
    tables = [
        pd.read_csv(folder / "table_a.csv"),
        pd.read_csv(folder / "table_b.csv"),
        pd.read_csv(folder / "table_c.csv"),
        pd.read_csv(folder / "table_d.csv"),
        pd.read_csv(folder / "table_e.csv"),
        pd.read_csv(folder / "table_pcr_a.csv"),
        pd.read_csv(folder / "table_pcr_b.csv"),
        pd.read_csv(folder / "table_pcr_c.csv"),
    ]
    letters = ["a", "b", "c", "d", "e", "a_pcr", "b_pcr", "c_pcr"]

    for i, table in enumerate(tables):
        table["sensitivity"] = table["Sensitivity [95% CI]"].apply(_extract_case_b_number(0))
        table["sensitivity_ci_lower"] = table["Sensitivity [95% CI]"].apply(_extract_case_b_number(1))
        table["sensitivity_ci_upper"] = table["Sensitivity [95% CI]"].apply(_extract_case_b_number(2))
        table["specificity"] = table["Specificity [95% CI]"].apply(_extract_case_b_number(0))
        table["specificity_ci_lower"] = table["Specificity [95% CI]"].apply(_extract_case_b_number(1))
        table["specificity_ci_upper"] = table["Specificity [95% CI]"].apply(_extract_case_b_number(2))
        table["type"] = letters[i]
        del table["Sensitivity [95% CI]"]
        del table["Specificity [95% CI]"]

    df = pd.concat(tables, ignore_index=True)
    x = df["sensitivity"].to_numpy(dtype=float).copy()
    y = df["specificity"].to_numpy(dtype=float).copy()
    rng = np.random.default_rng(seed)

    x_eq_1 = x == 1.0
    y_eq_1 = y == 1.0
    if np.any(x_eq_1):
        x[x_eq_1] -= rng.uniform(1e-6, eps_max, size=int(x_eq_1.sum()))
    if np.any(y_eq_1):
        y[y_eq_1] -= rng.uniform(1e-6, eps_max, size=int(y_eq_1.sum()))
    x = np.clip(x, 1e-12, 1 - 1e-12)
    y = np.clip(y, 1e-12, 1 - 1e-12)

    return Dataset(
        case_id="caseB",
        case_label="Case B: ANVISA COVID-19 diagnostic tests",
        x=x,
        y=y,
        metadata={
            "eps_max": float(eps_max),
            "jitter_seed": int(seed),
            "n_x_eq_1_jittered": int(x_eq_1.sum()),
            "n_y_eq_1_jittered": int(y_eq_1.sum()),
        },
    )


def sample_moments(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "m1": float(np.mean(x)),
        "m2": float(np.mean(y)),
        "v1": float(np.var(x, ddof=0)),
        "v2": float(np.var(y, ddof=0)),
        "rho": float(np.corrcoef(x, y)[0, 1]),
    }


def bivariate_beta_moments(alpha: np.ndarray) -> dict[str, float]:
    moments = BivariateBeta(np.asarray(alpha, dtype=float)).moments()
    return {
        "m1": float(moments[0]),
        "m2": float(moments[1]),
        "v1": float(moments[2]),
        "v2": float(moments[3]),
        "rho": float(moments[4]),
    }


def beta_mean_var(a: float, b: float) -> tuple[float, float]:
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return float(mean), float(var)


def independent_beta_moments(params: np.ndarray) -> dict[str, float]:
    a1, b1, a2, b2 = np.asarray(params, dtype=float)
    m1, v1 = beta_mean_var(a1, b1)
    m2, v2 = beta_mean_var(a2, b2)
    return {
        "m1": m1,
        "m2": m2,
        "v1": v1,
        "v2": v2,
        "rho": 0.0,
    }


def stabilize_dirichlet_alpha(alpha: np.ndarray, floor: float = DIRICHLET_ALPHA_FLOOR) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=float), float(floor))


def simulate_bivariate_beta(alpha: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.dirichlet(stabilize_dirichlet_alpha(alpha), size=int(size))
    return np.column_stack([u[:, 0] + u[:, 1], u[:, 0] + u[:, 2]]).astype(float)


def simulate_logit_normal(params: np.ndarray, size: int, seed: int) -> np.ndarray:
    mu1, mu2, sigma1, sigma2, rho = np.asarray(params, dtype=float)
    cov = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(mean=[mu1, mu2], cov=cov, size=int(size))
    return np.column_stack([expit(z[:, 0]), expit(z[:, 1])]).astype(float)


def simulate_independent_beta(params: np.ndarray, size: int, seed: int) -> np.ndarray:
    a1, b1, a2, b2 = np.asarray(params, dtype=float)
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.beta(a1, b1, size=int(size)),
            rng.beta(a2, b2, size=int(size)),
        ]
    ).astype(float)


def fit_logit_normal_closed_form_mle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.column_stack([logit(np.asarray(x, dtype=float)), logit(np.asarray(y, dtype=float))])
    mu = z.mean(axis=0)
    sigma = np.cov(z.T, ddof=0)

    sigma11 = max(float(sigma[0, 0]), LOGIT_NORMAL_SCALE_FLOOR)
    sigma22 = max(float(sigma[1, 1]), LOGIT_NORMAL_SCALE_FLOOR)
    sigma12 = float(sigma[0, 1])

    sigma1 = float(np.sqrt(sigma11))
    sigma2 = float(np.sqrt(sigma22))
    rho = float(np.clip(sigma12 / (sigma1 * sigma2), -LOGIT_NORMAL_RHO_CLIP, LOGIT_NORMAL_RHO_CLIP))

    return np.array([float(mu[0]), float(mu[1]), sigma1, sigma2, rho], dtype=float)


def build_stan_data_bln(x: np.ndarray, y: np.ndarray, use_priors: int, m: int = 0) -> dict:
    return {
        "n": int(x.shape[0]),
        "m": int(m),
        "xy": np.column_stack([x, y]).astype(float),
        "use_priors": int(use_priors),
        "s_sigma1": 2.0,
        "s_sigma2": 2.0,
        "tau": 5.0,
    }


def build_stan_data_bb(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(x.shape[0]),
        "xy": np.column_stack([x, y]).astype(float),
        "a": np.ones(4, dtype=float),
        "s_mu": 4.0,
        "s_sd": 1.0,
    }


def extract_bln_optimum(fit) -> np.ndarray:
    mu = np.asarray(fit.stan_variable("mu"), dtype=float)
    sigma1 = float(fit.stan_variable("sigma1"))
    sigma2 = float(fit.stan_variable("sigma2"))
    rho = float(fit.stan_variable("rho"))
    return np.array([mu[0], mu[1], sigma1, sigma2, rho], dtype=float)


def beta_mom_start(z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), eps, 1 - eps)
    mean = float(z.mean())
    var = float(z.var(ddof=0))
    scale = max(mean * (1 - mean) / max(var, eps) - 1, eps)
    return np.maximum(np.array([mean * scale, (1 - mean) * scale], dtype=float), eps)


def fit_beta_mle_1d(z: np.ndarray, init: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), eps, 1 - eps)
    init = beta_mom_start(z, eps=eps) if init is None else np.maximum(np.asarray(init, dtype=float), eps)

    def objective(log_params: np.ndarray) -> float:
        a, b = np.exp(log_params)
        loglik = ((a - 1) * np.log(z) + (b - 1) * np.log1p(-z)).sum() - len(z) * betaln(a, b)
        return -float(loglik)

    result = minimize(objective, x0=np.log(init), method="L-BFGS-B")
    if not result.success:
        fallback = minimize(objective, x0=np.log(np.array([1.0, 1.0])), method="L-BFGS-B")
        result = fallback if fallback.success else result
    if not result.success:
        raise RuntimeError(f"Beta MLE failed: {result.message}")
    return np.exp(result.x)


def fit_independent_beta_mle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a1, b1 = fit_beta_mle_1d(x)
    a2, b2 = fit_beta_mle_1d(y)
    return np.array([a1, b1, a2, b2], dtype=float)


def require_cmdstanpy(model_key: str) -> None:
    if CmdStanModel is None:
        raise ImportError(
            f"cmdstanpy is required for model '{model_key}' but is not installed in the current environment."
        )


@lru_cache(maxsize=None)
def get_bivariate_beta_model() -> CmdStanModel:
    require_cmdstanpy("beta_bayes")
    return CmdStanModel(stan_file=str(STAN_DIR / "bivariate-beta-model-lognormal.stan"), cpp_options={"STAN_THREADS": True})


@lru_cache(maxsize=None)
def get_logit_bivariate_model() -> CmdStanModel:
    require_cmdstanpy("bln_models")
    return CmdStanModel(stan_file=str(STAN_DIR / "logit-bivariate-model.stan"), cpp_options={"STAN_THREADS": True})


def energy_statistic_from_indices(
    distance_matrix: np.ndarray,
    row_sums: np.ndarray,
    total_sum: float,
    group_x_index: np.ndarray,
    variant: str,
) -> float:
    group_x_index = np.asarray(group_x_index, dtype=int)
    n = int(group_x_index.size)
    n_total = int(distance_matrix.shape[0])
    m = n_total - n
    if n <= 0 or m <= 0:
        raise ValueError("Both samples must be non-empty.")
    if variant == "u" and (n < 2 or m < 2):
        raise ValueError("The unbiased energy statistic requires both sample sizes to be at least 2.")

    s_xx = float(distance_matrix[np.ix_(group_x_index, group_x_index)].sum(dtype=np.float64))
    s_all_from_x = float(row_sums[group_x_index].sum(dtype=np.float64))
    s_xy = s_all_from_x - s_xx
    s_yy = float(total_sum - 2.0 * s_xy - s_xx)

    if variant == "v":
        within_x = s_xx / (n * n)
        within_y = s_yy / (m * m)
    elif variant == "u":
        within_x = s_xx / (n * (n - 1))
        within_y = s_yy / (m * (m - 1))
    else:
        raise ValueError(f"Unknown energy variant: {variant}")

    between = 2.0 * s_xy / (n * m)
    return float(between - within_x - within_y)


def energy_two_sample_test(
    observed_xy: np.ndarray,
    reference_xy: np.ndarray,
    n_permutations: int,
    seed: int,
    variant: str,
) -> dict[str, float]:
    observed_xy = np.asarray(observed_xy, dtype=float)
    reference_xy = np.asarray(reference_xy, dtype=float)
    if observed_xy.ndim != 2 or reference_xy.ndim != 2:
        raise ValueError("Samples must be two-dimensional arrays.")
    if observed_xy.shape[1] != reference_xy.shape[1]:
        raise ValueError("Samples must have the same feature dimension.")

    pooled = np.vstack([observed_xy, reference_xy]).astype(float)
    distance_matrix = cdist(pooled, pooled, metric="euclidean").astype(np.float32, copy=False)
    row_sums = distance_matrix.sum(axis=1, dtype=np.float64)
    total_sum = float(row_sums.sum())
    n_obs = observed_xy.shape[0]
    n_total = pooled.shape[0]

    obs_index = np.arange(n_obs, dtype=int)
    obs_stat = energy_statistic_from_indices(distance_matrix, row_sums, total_sum, obs_index, variant=variant)

    rng = np.random.default_rng(seed)
    perm_stats = np.empty(int(n_permutations), dtype=float)
    for b in trange(int(n_permutations), desc="Energy permutations", leave=False):
        idx_x = rng.choice(n_total, size=n_obs, replace=False)
        perm_stats[b] = energy_statistic_from_indices(distance_matrix, row_sums, total_sum, idx_x, variant=variant)

    p_value = (1.0 + float(np.sum(perm_stats >= obs_stat))) / (n_permutations + 1.0)
    perm_mean = float(np.mean(perm_stats))
    perm_sd = float(np.std(perm_stats, ddof=1)) if perm_stats.size > 1 else np.nan
    z_score = float((obs_stat - perm_mean) / (perm_sd + 1e-12)) if np.isfinite(perm_sd) else np.nan
    p_value_mcse = float(np.sqrt(p_value * (1.0 - p_value) / (n_permutations + 1.0)))
    return {
        "statistic": float(obs_stat),
        "p_value": float(p_value),
        "perm_mean": perm_mean,
        "perm_sd": perm_sd,
        "z_score": z_score,
        "p_value_mcse": p_value_mcse,
    }


def fit_model(
    model_key: str,
    dataset: Dataset,
    args: argparse.Namespace,
    distribution: BivariateBeta,
    shared_fit_cache: dict | None = None,
) -> ModelResult:
    case_offset = CASE_SEED_OFFSET[dataset.case_id]
    model_offset = MODEL_SEED_OFFSET[model_key]
    fit_seed = int(args.seed + case_offset + model_offset)
    sim_seed = int(args.seed + case_offset + model_offset + 1_000_000)

    if model_key == "mom_closed":
        alpha = distribution.method_moments_estimator_1(dataset.x, dataset.y)
        ref = simulate_bivariate_beta(alpha, size=args.reference_size, seed=sim_seed)
        return ModelResult(
            model_key=model_key,
            model_label=MODEL_LABEL[model_key],
            parameter_names=("alpha_1", "alpha_2", "alpha_3", "alpha_4"),
            parameter_values=np.asarray(alpha, dtype=float),
            reference_size=args.reference_size,
            reference_sample=ref,
            fit_metadata={
                "estimator": "method_moments_estimator_1",
                "simulation_alpha_floor": DIRICHLET_ALPHA_FLOOR,
                "simulation_alpha_clipped": bool(np.any(np.asarray(alpha, dtype=float) <= DIRICHLET_ALPHA_FLOOR)),
            },
        )

    if model_key == "mom_nls":
        alpha = distribution.method_moments_estimator_4(dataset.x, dataset.y)
        ref = simulate_bivariate_beta(alpha, size=args.reference_size, seed=sim_seed)
        return ModelResult(
            model_key=model_key,
            model_label=MODEL_LABEL[model_key],
            parameter_names=("alpha_1", "alpha_2", "alpha_3", "alpha_4"),
            parameter_values=np.asarray(alpha, dtype=float),
            reference_size=args.reference_size,
            reference_sample=ref,
            fit_metadata={
                "estimator": "method_moments_estimator_4",
                "simulation_alpha_floor": DIRICHLET_ALPHA_FLOOR,
                "simulation_alpha_clipped": bool(np.any(np.asarray(alpha, dtype=float) <= DIRICHLET_ALPHA_FLOOR)),
            },
        )

    if model_key == "ind_beta_mle":
        params = fit_independent_beta_mle(dataset.x, dataset.y)
        ref = simulate_independent_beta(params, size=args.reference_size, seed=sim_seed)
        return ModelResult(
            model_key=model_key,
            model_label=MODEL_LABEL[model_key],
            parameter_names=("a1", "b1", "a2", "b2"),
            parameter_values=np.asarray(params, dtype=float),
            reference_size=args.reference_size,
            reference_sample=ref,
            fit_metadata={"estimator": "independent_beta_mle"},
        )

    if model_key == "mle":
        params = fit_logit_normal_closed_form_mle(dataset.x, dataset.y)
        ref = simulate_logit_normal(params, size=args.reference_size, seed=sim_seed)
        return ModelResult(
            model_key=model_key,
            model_label=MODEL_LABEL[model_key],
            parameter_names=("mu_1", "mu_2", "sigma_1", "sigma_2", "rho"),
            parameter_values=np.asarray(params, dtype=float),
            reference_size=args.reference_size,
            reference_sample=ref,
            fit_metadata={
                "estimator": "closed-form logistic-normal MLE",
                "use_priors": 0,
                "covariance_ddof": 0,
            },
        )

    if model_key == "bln_bayes":
        model = get_logit_bivariate_model()
        data = build_stan_data_bln(dataset.x, dataset.y, use_priors=1, m=0)
        fit = model.sample(
            data=data,
            chains=args.chains,
            parallel_chains=min(args.chains, os.cpu_count() or 1),
            iter_sampling=args.bln_draws,
            iter_warmup=args.bln_warmup,
            seed=fit_seed,
            show_progress=True,
        )
        mu_draws = np.asarray(fit.stan_variable("mu"), dtype=float)
        sigma1_draws = np.asarray(fit.stan_variable("sigma1"), dtype=float).reshape(-1, 1)
        sigma2_draws = np.asarray(fit.stan_variable("sigma2"), dtype=float).reshape(-1, 1)
        rho_draws = np.asarray(fit.stan_variable("rho"), dtype=float).reshape(-1, 1)
        draws = np.hstack([mu_draws, sigma1_draws, sigma2_draws, rho_draws])
        params = posterior_point_estimate(draws, args.bayes_point_estimate)
        ref = simulate_logit_normal(params, size=args.reference_size, seed=sim_seed)
        return ModelResult(
            model_key=model_key,
            model_label=MODEL_LABEL[model_key],
            parameter_names=("mu_1", "mu_2", "sigma_1", "sigma_2", "rho"),
            parameter_values=np.asarray(params, dtype=float),
            reference_size=args.reference_size,
            reference_sample=ref,
            fit_metadata={
                "estimator": f"CmdStan sample ({args.bayes_point_estimate})",
                "use_priors": 1,
                "chains": int(args.chains),
                "iter_sampling": int(args.bln_draws),
                "iter_warmup": int(args.bln_warmup),
            },
        )

    if model_key in {"beta_bayes", "beta_bayes_mean", "beta_bayes_median"}:
        cache_key = f"{dataset.case_id}_beta_bayes_fit"
        fit = None
        adapt_delta = (
            args.bb_adapt_delta_case_a
            if dataset.case_id == "caseA"
            else args.bb_adapt_delta_case_b
        )
        if shared_fit_cache is not None:
            fit = shared_fit_cache.get(cache_key)
        if fit is None:
            model = get_bivariate_beta_model()
            data = build_stan_data_bb(dataset.x, dataset.y)
            fit = model.sample(
                data=data,
                chains=args.chains,
                parallel_chains=min(args.chains, os.cpu_count() or 1),
                iter_sampling=args.bb_draws,
                iter_warmup=args.bb_warmup,
                adapt_delta=adapt_delta,
                seed=fit_seed,
                show_progress=True,
            )
            if shared_fit_cache is not None:
                shared_fit_cache[cache_key] = fit
        alpha_draws = np.asarray(fit.stan_variable("alpha"), dtype=float)
        if model_key == "beta_bayes_mean":
            point_estimate = "mean"
        elif model_key == "beta_bayes_median":
            point_estimate = "median"
        else:
            point_estimate = args.bayes_point_estimate
        alpha = posterior_point_estimate(alpha_draws, point_estimate)
        ref = simulate_bivariate_beta(alpha, size=args.reference_size, seed=sim_seed)
        return ModelResult(
            model_key=model_key,
            model_label=MODEL_LABEL[model_key],
            parameter_names=("alpha_1", "alpha_2", "alpha_3", "alpha_4"),
            parameter_values=np.asarray(alpha, dtype=float),
            reference_size=args.reference_size,
            reference_sample=ref,
            fit_metadata={
                "estimator": f"CmdStan sample ({point_estimate})",
                "prior_family": "lognormal",
                "theta_prior": [1.0, 1.0, 1.0, 1.0],
                "s_prior": {"family": "lognormal", "mu_log_scale": 4.0, "sd_log_scale": 1.0},
                "chains": int(args.chains),
                "iter_sampling": int(args.bb_draws),
                "iter_warmup": int(args.bb_warmup),
                "adapt_delta": float(adapt_delta),
                "simulation_alpha_floor": DIRICHLET_ALPHA_FLOOR,
                "simulation_alpha_clipped": bool(np.any(np.asarray(alpha, dtype=float) <= DIRICHLET_ALPHA_FLOOR)),
            },
        )

    raise ValueError(f"Unsupported model key: {model_key}")


def parameter_dict(parameter_names: tuple[str, ...], parameter_values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(parameter_names, np.asarray(parameter_values, dtype=float))
    }


def implied_moments_for_model(model_key: str, parameter_values: np.ndarray, reference_sample: np.ndarray) -> dict[str, float]:
    if model_key in {"mom_closed", "mom_nls", "beta_bayes", "beta_bayes_mean", "beta_bayes_median"}:
        return bivariate_beta_moments(parameter_values)
    if model_key == "ind_beta_mle":
        return independent_beta_moments(parameter_values)
    return sample_moments(reference_sample[:, 0], reference_sample[:, 1])


def run_case(
    dataset: Dataset,
    model_keys: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    distribution = BivariateBeta()
    shared_fit_cache: dict = {}
    observed_xy = dataset.xy
    observed_mom = sample_moments(dataset.x, dataset.y)
    rows: list[dict] = []

    for model_key in model_keys:
        print(f"[{dataset.case_id}] fitting {model_key}")
        result = fit_model(model_key, dataset, args, distribution, shared_fit_cache=shared_fit_cache)

        print(f"[{dataset.case_id}] energy test for {model_key}")
        perm_seed = int(args.seed + CASE_SEED_OFFSET[dataset.case_id] + MODEL_SEED_OFFSET[model_key] + 2_000_000)
        energy_summary = energy_two_sample_test(
            observed_xy=observed_xy,
            reference_xy=result.reference_sample,
            n_permutations=args.permutations,
            seed=perm_seed,
            variant=args.energy_variant,
        )

        param_json = json.dumps(parameter_dict(result.parameter_names, result.parameter_values), sort_keys=True)
        model_implied = implied_moments_for_model(result.model_key, result.parameter_values, result.reference_sample)
        ref_mom = sample_moments(result.reference_sample[:, 0], result.reference_sample[:, 1])

        row = {
            "case_id": dataset.case_id,
            "case_label": dataset.case_label,
            "model": result.model_key,
            "model_label": result.model_label,
            "n_observed": int(dataset.n),
            "reference_size_M": int(result.reference_size),
            "energy_variant": args.energy_variant,
            "energy_statistic": float(energy_summary["statistic"]),
            "p_value": float(energy_summary["p_value"]),
            "perm_null_mean": float(energy_summary["perm_mean"]),
            "perm_null_sd": float(energy_summary["perm_sd"]),
            "energy_z_score": float(energy_summary["z_score"]),
            "p_value_mcse": float(energy_summary["p_value_mcse"]),
            "parameter_json": param_json,
            "fit_metadata_json": json.dumps(result.fit_metadata, sort_keys=True),
            "obs_m1": observed_mom["m1"],
            "obs_m2": observed_mom["m2"],
            "obs_v1": observed_mom["v1"],
            "obs_v2": observed_mom["v2"],
            "obs_rho": observed_mom["rho"],
            "fit_m1": model_implied["m1"],
            "fit_m2": model_implied["m2"],
            "fit_v1": model_implied["v1"],
            "fit_v2": model_implied["v2"],
            "fit_rho": model_implied["rho"],
            "ref_m1": ref_mom["m1"],
            "ref_m2": ref_mom["m2"],
            "ref_v1": ref_mom["v1"],
            "ref_v2": ref_mom["v2"],
            "ref_rho": ref_mom["rho"],
        }
        for name, value in parameter_dict(result.parameter_names, result.parameter_values).items():
            row[name] = value
        rows.append(row)

    table = pd.DataFrame(rows)
    table["model"] = pd.Categorical(table["model"], categories=model_keys, ordered=True)
    table = table.sort_values("model").reset_index(drop=True)
    table["model"] = table["model"].astype(str)
    metadata = {
        "case_id": dataset.case_id,
        "case_label": dataset.case_label,
        "dataset_metadata": dataset.metadata,
        "models": model_keys,
        "reference_size": int(args.reference_size),
        "permutations": int(args.permutations),
        "energy_variant": args.energy_variant,
        "bayes_point_estimate": args.bayes_point_estimate,
        "seed": int(args.seed),
    }
    return table, metadata


def default_models_for_case(case_id: str) -> list[str]:
    if case_id == "caseA":
        return CASE_A_DEFAULT_MODELS.copy()
    if case_id == "caseB":
        return CASE_B_DEFAULT_MODELS.copy()
    raise ValueError(f"Unknown case id: {case_id}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = ["caseA", "caseB"] if args.case == "all" else [args.case]
    dataset_lookup = {
        "caseA": load_case_a(year=args.case_a_year, tie_epsilon=args.case_a_tie_epsilon, seed=args.case_a_seed),
        "caseB": load_case_b(seed=args.case_b_seed, eps_max=args.case_b_eps_max),
    }

    summary_records = []
    for case_id in cases:
        dataset = dataset_lookup[case_id]
        model_keys = args.models if args.models is not None else default_models_for_case(case_id)
        table, metadata = run_case(dataset, model_keys, args)

        csv_path = args.output_dir / f"{case_id}_energy_gof_M{args.reference_size}_B{args.permutations}.csv"
        json_path = args.output_dir / f"{case_id}_energy_gof_M{args.reference_size}_B{args.permutations}.json"
        table.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        print()
        print(f"Saved {case_id} results to {csv_path}")
        print(table[["model", "reference_size_M", "energy_statistic", "p_value"]].round(6).to_string(index=False))
        print()

        summary_records.append(
            {
                "case_id": case_id,
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "n_models": int(table.shape[0]),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_path = args.output_dir / f"energy_gof_summary_M{args.reference_size}_B{args.permutations}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary index to {summary_path}")


if __name__ == "__main__":
    main()
