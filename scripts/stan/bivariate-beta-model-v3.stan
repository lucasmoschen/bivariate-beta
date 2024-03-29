// Bivariate beta model 
// Estimate the parameter alpha 
// The variable U is used as a latent variable with uniform prior.

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
}
data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;
   // gamma prior
   vector<lower=0>[4] a;
   vector<lower=0>[4] b;
}
transformed data {
   vector[n] lb;
   vector[n] ub;
   for (i in 1:n) {
      lb[i] = max({0.0, xy[i,1]+xy[i,2]-1});
      ub[i] = min(xy[i]);
   }
}
parameters {
   vector<lower=0>[4] alpha;
   vector<lower=0, upper=1>[n] u_raw;
}
transformed parameters {
   vector[n] u = (ub - lb) .* u_raw + lb;
}
model {
    alpha ~ gamma(a, b);
    u_raw ~ uniform(0, 1);
    xy ~ log_bivariate_beta(alpha, u, n);
}
