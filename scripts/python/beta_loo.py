import numpy as np
import arviz as az

from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, logsumexp


def build_beta_loglik_cache(x, y, n_quad=48, eps=1e-12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    nodes, weights = leggauss(int(n_quad))
    t = 0.5 * (nodes + 1.0)
    w = 0.5 * weights

    lb = np.maximum(0.0, x + y - 1.0)
    ub = np.minimum(x, y)
    width = ub - lb

    u = lb[:, None] + width[:, None] * t[None, :]
    x_minus_u = x[:, None] - u
    y_minus_u = y[:, None] - u
    one_minus_x_minus_y_plus_u = 1.0 - x[:, None] - y[:, None] + u

    # Endpoint clipping only stabilizes the quadrature numerically; it does not
    # change the interior nodes.
    safe = lambda arr: np.maximum(arr, eps)

    return {
        "n_obs": int(x.size),
        "log_weight_width": np.log(safe(width))[:, None] + np.log(w)[None, :],
        "log_u": np.log(safe(u)),
        "log_x_minus_u": np.log(safe(x_minus_u)),
        "log_y_minus_u": np.log(safe(y_minus_u)),
        "log_one_minus_x_minus_y_plus_u": np.log(safe(one_minus_x_minus_y_plus_u)),
    }


def _subsample_alpha_draws(alpha_draws, take=None, seed=12345):
    alpha_draws = np.asarray(alpha_draws, dtype=float)
    if alpha_draws.ndim != 2 or alpha_draws.shape[1] != 4:
        raise ValueError("alpha_draws must have shape (n_draws, 4).")
    if take is not None and take < len(alpha_draws):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(alpha_draws), size=int(take), replace=False)
        alpha_draws = alpha_draws[idx]
    return alpha_draws


def beta_marginal_loglik_matrix(
    alpha_draws,
    *,
    x=None,
    y=None,
    cache=None,
    take=None,
    seed=12345,
    n_quad=48,
    eps=1e-12,
    batch_size=256,
):
    alpha_draws = _subsample_alpha_draws(alpha_draws, take=take, seed=seed)
    if cache is None:
        if x is None or y is None:
            raise ValueError("Either cache or both x and y must be provided.")
        cache = build_beta_loglik_cache(x, y, n_quad=n_quad, eps=eps)

    n_draws = alpha_draws.shape[0]
    n_obs = cache["n_obs"]
    out = np.empty((1, n_draws, n_obs), dtype=float)

    log_norm = gammaln(alpha_draws).sum(axis=1) - gammaln(alpha_draws.sum(axis=1))

    log_weight_width = cache["log_weight_width"]
    log_u = cache["log_u"]
    log_x_minus_u = cache["log_x_minus_u"]
    log_y_minus_u = cache["log_y_minus_u"]
    log_rest = cache["log_one_minus_x_minus_y_plus_u"]

    for start in range(0, n_draws, int(batch_size)):
        stop = min(start + int(batch_size), n_draws)
        alpha_batch = alpha_draws[start:stop]
        kernel = (
            log_weight_width[None, :, :]
            + (alpha_batch[:, 0, None, None] - 1.0) * log_u[None, :, :]
            + (alpha_batch[:, 1, None, None] - 1.0) * log_x_minus_u[None, :, :]
            + (alpha_batch[:, 2, None, None] - 1.0) * log_y_minus_u[None, :, :]
            + (alpha_batch[:, 3, None, None] - 1.0) * log_rest[None, :, :]
        )
        out[0, start:stop, :] = logsumexp(kernel, axis=2) - log_norm[start:stop, None]

    return out


def beta_bayes_loo(
    alpha_draws,
    *,
    x=None,
    y=None,
    cache=None,
    take=None,
    seed=12345,
    n_quad=48,
    eps=1e-12,
    batch_size=256,
):
    alpha_draws = _subsample_alpha_draws(alpha_draws, take=take, seed=seed)
    log_lik = beta_marginal_loglik_matrix(
        alpha_draws,
        x=x,
        y=y,
        cache=cache,
        take=None,
        seed=seed,
        n_quad=n_quad,
        eps=eps,
        batch_size=batch_size,
    )
    loo_idata = az.from_dict(
        posterior={"alpha": alpha_draws[None, :, :]},
        log_likelihood={"obs": log_lik},
        coords={
            "alpha_dim": np.arange(alpha_draws.shape[1]),
            "obs_id": np.arange(log_lik.shape[-1]),
        },
        dims={
            "alpha": ["alpha_dim"],
            "obs": ["obs_id"],
        },
    )
    return az.loo(loo_idata, var_name="obs", pointwise=True)


def beta_bayes_loo_from_model_fit(
    model_fit,
    *,
    x=None,
    y=None,
    cache=None,
    take=None,
    seed=12345,
    n_quad=48,
    eps=1e-12,
    batch_size=256,
):
    alpha_draws = np.asarray(model_fit.stan_variable("alpha"), dtype=float)
    return beta_bayes_loo(
        alpha_draws,
        x=x,
        y=y,
        cache=cache,
        take=take,
        seed=seed,
        n_quad=n_quad,
        eps=eps,
        batch_size=batch_size,
    )
