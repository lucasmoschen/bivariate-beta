// Bivariate beta model 
// Estimate the parameter alpha 
// The variable U is used as a latent variable with uniform prior.

functions {
   real log_multi_beta(vector theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data matrix xy, vector alpha, vector u_raw, data int n, data vector lb, data vector ub){
      vector[n] u = (ub - lb) .* u_raw + lb;
      vector[n] x = col(xy, 1);
      vector[n] y = col(xy, 2);
      real v = sum(lmultiply(alpha[1]-1, fmax(u, 1e-12)) + lmultiply(alpha[2]-1, fmax(x-u, 1e-12)));
      v += sum(lmultiply(alpha[3]-1, fmax(y-u, 1e-12)) + (alpha[4]-1) * log1m(fmin(1-1e-12, x+y-u)));
      v += -n * log_multi_beta(alpha);
      return v;
   }
   real log_bivariate_beta_obs_lpdf(vector xy, real u_raw,
                                    vector alpha, real lb, real ub) {
      real x = xy[1];
      real y = xy[2];
      real u = (ub - lb) * u_raw + lb;
      real v = (alpha[1]-1) * log(fmax(u, 1e-12))
             + (alpha[2]-1) * log(fmax(x-u, 1e-12))
             + (alpha[3]-1) * log(fmax(y-u, 1e-12))
             + (alpha[4]-1) * log1m(fmin(1-1e-12, x+y-u));
      v += -log_multi_beta(alpha);
      return v;
   }
}
data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;
   // dirichlet prior
   vector<lower=0>[4] a;
   real<lower=0> kappa;
   real<lower=0> lambda;
}
transformed data {
   vector[n] lb;
   vector[n] ub;
   for (i in 1:n) {
      lb[i] = fmax(0.0, xy[i,1]+xy[i,2]-1);
      ub[i] = fmin(xy[i,1], xy[i,2]);
   }
}
parameters {
   real<lower=0> s;
   simplex[4] theta;
   vector<lower=0, upper=1>[n] u_raw;
}
transformed parameters {
   vector[4] alpha = s * theta;
}
model {
   s ~ gamma(kappa, lambda); 
   theta ~ dirichlet(a);
   u_raw ~ uniform(0, 1);
   xy ~ log_bivariate_beta(alpha, u_raw, n, lb, ub);
}
generated quantities {
  vector[n] log_lik;
  for (i in 1:n) {
    log_lik[i] = log_bivariate_beta_obs_lpdf(to_vector(xy[i]) | u_raw[i], alpha, lb[i], ub[i]);
  }

  matrix[n,2] xy_tilde;
  for (j in 1:n) {
    vector[4] u = dirichlet_rng(alpha);
    xy_tilde[j,1] = u[1] + u[2];
    xy_tilde[j,2] = u[1] + u[3];
  }
}