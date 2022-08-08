data {
    real s;
    real<lower=0> alpha1;
    real<lower=0> alpha2;
    real<lower=0> beta1;
    real<lower=0> beta2;
}
parameters {
    real<lower=0, upper=s> x;
}
model {
    target += gamma_lpdf(x | alpha1, beta1);
    target += gamma_lpdf(s-x| alpha2, beta2);
}
