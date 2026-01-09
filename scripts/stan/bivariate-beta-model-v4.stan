// Bivariate beta model via (m1, m2, s, rho) reparameterization
// Latent variable u implements the Dirichlet integral representation.

functions {
   real log_multi_beta(vector theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data matrix xy, vector alpha, vector u, data int n){
      vector[n] x = col(xy, 1);
      vector[n] y = col(xy, 2);
      real v = sum(lmultiply(alpha[1]-1, u) + lmultiply(alpha[2]-1, x-u));
      v += sum(lmultiply(alpha[3]-1, y-u) + (alpha[4]-1) * log1m(x+y-u));
      v += -n * log_multi_beta(alpha);
      return v;
   }
    vector alpha_from_mmsr(real m1, real m2, real s, real rho) {
      real sq = sqrt(m1 * m2 * (1 - m1) * (1 - m2));
      vector[4] a;
      a[4] = s * (rho * sq + (1 - m1) * (1 - m2));
      a[1] = (m1 + m2 - 1) * s + a[4];
      a[2] = (1 - m2) * s - a[4];
      a[3] = (1 - m1) * s - a[4];
      return a;
  }
  vector rho_bounds(real m1, real m2) {
    real num_lo = -fmin(m1 * m2, (1 - m1) * (1 - m2));
    real num_hi =  fmin(m1, m2) - m1 * m2;
    real sq = sqrt(m1 * m2 * (1 - m1) * (1 - m2));
    vector[2] b;
    b[1] = num_lo / sq;
    b[2] = num_hi / sq;
    return b;
  }
  real project_rho(real r_tilde, real rho_min, real rho_max, real nu) {
    real lo = rho_min*nu + rho_max*(1-nu);
    real hi = rho_min*(1-nu) + rho_max*nu;
    return 0.5 * (hi + lo) + 0.5 * (hi - lo) * tanh(r_tilde);
  }
}

data {
  int<lower=1> n;
  matrix<lower=0, upper=1>[n, 2] xy;

  // m1, m2 ~ Beta(a_m, b_m)
  real<lower=0> a_m1;
  real<lower=0> b_m1;
  real<lower=0> a_m2;
  real<lower=0> b_m2;

  // s ~ Gamma(a_s, b_s)
  real<lower=0> a_s;
  real<lower=0> b_s;

  // r_tilde ~ Normal(mu_r, sigma_r)
  real mu_r;
  real<lower=0> sigma_r;

  real<lower=0.5000001, upper=0.9999999> nu;
}

transformed data {
  vector[n] lb;
  vector[n] ub;
  for (i in 1:n) {
    lb[i] = fmax(0.0, xy[i,1] + xy[i,2] - 1.0);
    ub[i] = fmin(xy[i,1], xy[i,2]);
  }
}

parameters {
  real<lower=0,upper=1> m1;
  real<lower=0,upper=1> m2;
  real<lower=0> s;
  real r_tilde;
  vector<lower=0, upper=1>[n] u_raw;
}

transformed parameters {
  vector[2] rb = rho_bounds(m1, m2);
  real rho = project_rho(r_tilde, rb[1], rb[2], nu);

  vector[4] alpha = alpha_from_mmsr(m1, m2, s, rho);
  vector[n] u = lb + (ub - lb) .* u_raw;
}

model {
  m1 ~ beta(a_m1, b_m1);
  m2 ~ beta(a_m2, b_m2);
  s  ~ gamma(a_s, b_s);

  r_tilde ~ normal(mu_r, sigma_r);
  u_raw ~ uniform(0, 1);

  target += log_bivariate_beta_lpdf(xy | alpha, u, n);
}

generated quantities {
  real vx = (alpha[1] + alpha[2]) * (alpha[3] + alpha[4])
            / square(sum(alpha)) / (sum(alpha) + 1);
  real vy = (alpha[1] + alpha[3]) * (alpha[2] + alpha[4])
            / square(sum(alpha)) / (sum(alpha) + 1);
  real cov_xy = (alpha[1] * alpha[4] - alpha[2] * alpha[3])
                / square(sum(alpha)) / (sum(alpha) + 1);
  real corr_xy = cov_xy / sqrt(vx * vy);
}
