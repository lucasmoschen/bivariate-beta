// Bivariate beta model 
// Estimate the parameter alpha and U
// Here alpha is treated in the log scale.

functions {
   real log_multi_beta(vector theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data matrix xy, vector alpha_log, vector u, data int n){
      vector[n] x = col(xy, 1);
      vector[n] y = col(xy, 2);
      real v = expm1(alpha_log[1]) * sum(log(u)) + expm1(alpha_log[2]) * sum(log(x-u));
      v += expm1(alpha_log[3]) * sum(log(y-u)) + expm1(alpha_log[4]) * sum(log1m(x+y-u));
      v += -n * log_multi_beta(exp(alpha_log));
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
   vector[4] alpha_log;
   vector<lower=0, upper=1>[n] u_raw;
}
transformed parameters {
   vector[n] u = (ub - lb) .* u_raw + lb;
}
model {
    // prior
    target += a .* alpha_log - b .* exp(alpha_log);
    target += log_bivariate_beta_lpdf(xy | alpha_log, u, n);
}
