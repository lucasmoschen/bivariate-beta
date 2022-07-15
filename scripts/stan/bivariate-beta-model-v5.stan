functions {
   real log_multi_beta(array[] real theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_transformed_lpdf(data vector x, data vector y, vector z1, vector z3, array[] real alpha, data int n){
      real v = (alpha[1]-1)*sum(log(x+y+z1-1));
      v += (alpha[3]+alpha[4]-1)*sum(log1m(x));
      v += (alpha[2]-1)*sum(log1m(y+z1));
      v += (alpha[4]-1)*sum(log(z3));
      v += (alpha[3]-1)*sum(log1m(z3));
      v += -n * log_multi_beta(alpha);
      return v;
   }
}
data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;
   // gamma prior
   array[4] real<lower=0> a;
   array[4] real<lower=0> b;
}
transformed data {
   vector[n] x = col(xy, 1);
   vector[n] y = col(xy, 2);
   vector[n] lb;
   vector[n] ub;
   for (i in 1:n) {
      lb[i] = max({(1 - x[i] - y[i])/(1-x[i]), 0.0});
      ub[i] = min({(1-y[i])/(1-x[i]), 1.0});
   }
}
parameters {
   array[4] real<lower=0> alpha;
   vector<lower=lb, upper=ub>[n] z3;
}
transformed parameters {
    //vector<lower=0, upper=1>[n] z1 = (1-x) .* z3;
    vector<lower=0, upper=1>[n] z1 = 1-y + (1-x) .* (1-z3);
    vector[n] z2 = (1-x)./z1;
}
model {
    // prior
    target += gamma_lpdf(alpha | a, b);
    // model
    target += beta_lpdf(z1 | sum(alpha[2:4]), alpha[1]);
    target += beta_lpdf(z2 | sum(alpha[3:4]), alpha[2]);
    target += beta_lpdf(z3 | alpha[4], alpha[3]);
    // jacobian
    target += -sum(log(z1));
    //target += log_bivariate_beta_transformed_lpdf(x | y, z1, z3, alpha, n);
}