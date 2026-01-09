// Bivariate beta model 
// Estimate the parameter alpha 
// The variable U is used as a latent variable with uniform prior.
data {
    int<lower=1> n; // number of observations
    int<lower=0> m; // number of observations
    matrix[n,2] xy; // data
    real<lower=0> s_sigma1;
    real<lower=0> s_sigma2;
    real<lower=0> tau;
    int<lower=0,upper=1> use_priors;
}
transformed data {
    matrix[n,2] logit_xy;
    for (i in 1:n) {
        logit_xy[i,1] = logit(xy[i,1]);
        logit_xy[i,2] = logit(xy[i,2]);
    }
}
parameters {
    vector[2] mu ;
    real<lower=1e-6> sigma1;
    real<lower=1e-6> sigma2;
    real<lower=-0.99, upper=0.99> rho;
}
transformed parameters {
    cov_matrix[2] Sigma;
    Sigma[1,1] = square(sigma1);
    Sigma[2,2] = square(sigma2);
    Sigma[1,2] = rho * sigma1 * sigma2;
    Sigma[2,1] = Sigma[1,2];
}
model {
    if (use_priors == 1) {
        sigma1 ~ normal(1e-6, s_sigma1);
        sigma2 ~ normal(1e-6, s_sigma2);
        mu ~ normal(0, tau);
        rho ~ uniform(-0.99, 0.99);
    }
    for (i in 1:n) {
        logit_xy[i] ~ multi_normal(mu, Sigma);
    }
}
generated quantities {
    matrix[m,2] xy_tilde;
    for (i in 1:m) {
        xy_tilde[i] = to_row_vector(inv_logit(multi_normal_rng(mu, Sigma)));
    } 
    vector[n] log_lik;
    for (i in 1:n) {
        log_lik[i] = multi_normal_lpdf(to_vector(logit_xy[i]) | mu, Sigma) - log(xy[i,1]) - log1m(xy[i,1]) - log(xy[i,2]) - log1m(xy[i,2]);
    }
}