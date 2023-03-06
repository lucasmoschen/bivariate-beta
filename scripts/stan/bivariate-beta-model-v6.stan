// Bivariate beta model 
// Estimate the latent variable U
// The alpha parameter is fixed and for U is used the same model as v3.

functions {
   real log_multi_beta(array[] real theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data matrix xy, vector u, data int n, 
                                data real alpha1, data real alpha2, data real alpha3, data real alpha4){
      vector[n] x = col(xy, 1);
      vector[n] y = col(xy, 2);
      real v = (alpha1-1) * sum(log(u)) + (alpha2-1) * sum(log(x-u));
      v += (alpha3-1) * sum(log(y-u)) + (alpha4-1) * sum(log1m(x+y-u));
      v += -n * log_multi_beta({alpha1, alpha2, alpha3, alpha4});
      return v;
   }
}
data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;

   real<lower=0> alpha1;
   real<lower=0> alpha2;
   real<lower=0> alpha3;
   real<lower=0> alpha4;
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
   vector<lower=0, upper=1>[n] u_raw;
}
transformed parameters {
   vector[n] u = (ub - lb) .* u_raw + lb;
}
model {
    u_raw ~ uniform(0, 1);
    xy ~ log_bivariate_beta(u, n, alpha1, alpha2, alpha3, alpha4);
}
