// Bivariate beta model 
// Estimate the parameter alpha 
// The variable U is used as a latent variable with uniform prior.

functions {
   real log_multi_beta(vector theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
}
data {
   int<lower=0> n;
   vector<lower=0, upper=1>[n] x;
   vector<lower=0, upper=1>[n] y;
   // gamma prior
   vector<lower=0>[4] a;
   vector<lower=0>[4] b;
}
transformed data {
    vector[n] lb;
    vector[n] ub;
    for (i in 1:n) {
        lb[i] = max({0.0, x[i]+y[i]-1.0});
        ub[i] = min({x[i], y[i]});
    }
    vector[n] one = rep_vector(1.0, n);
}
parameters {
    vector<lower=0>[4] alpha;
    vector<lower=lb, upper=ub>[n] u;
}
transformed parameters {
   vector<lower=0, upper=1>[n] u2 = x-u;
   vector<lower=0, upper=1>[n] u3 = y-u;
   vector<lower=0, upper=1>[n] u4 = 1-x-y+u;
}
model {
    target += gamma_lpdf(alpha | a, b);
    target += (alpha[1]-1) * sum(log(u));//- log(one./u + 1.0)); 
    target += (alpha[2]-1) * sum(log(u2));// - log(one./u2 + 1.0));
    target += (alpha[3]-1) * sum(log(u3));// - log(one./u3 + 1.0));
    target += (alpha[4]-1) * sum(log(u4));// - log(one./u4 + 1.0));
    target += -n * log_multi_beta(alpha);
}
