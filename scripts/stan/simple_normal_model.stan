data {
    real s;
    real mu1;
    real mu2;
    real<lower=0> sigma1;
    real<lower=0> sigma2;
}
parameters {
    real x;
}
model {
    x ~ normal(mu1, sigma1);
    s ~ normal(mu2 + x, sigma2);
}
