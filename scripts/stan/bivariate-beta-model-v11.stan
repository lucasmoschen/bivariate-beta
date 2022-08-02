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
        lb[i] = max({1.0-x[i]-y[i], -1.0+x[i]+y[i]});
        ub[i] = min({1.0+x[i]-y[i], 1.0-x[i]+y[i]});
    }
}
parameters {
    vector<lower=0>[4] alpha;
    vector<lower=lb, upper=ub>[n] z;
}
model {
    target += gamma_lpdf(alpha | a, b);
    target += (alpha[1]-1) * log(x+y+z-1); 
    target += (alpha[2]-1) * log1p(x-y-z);
    target += (alpha[3]-1) * log1p(-x+y-z);
    target += (alpha[4]-1) * log1p(-x-y+z);
    target += -n * ((sum(alpha)-3) * log(2) + log_multi_beta(alpha));
}
