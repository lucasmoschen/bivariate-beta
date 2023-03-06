// Bivariate beta model 
// Estimate the parameter alpha 
// The variable U is fixed here.

functions {
   real log_multi_beta(vector theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data matrix xy, vector alpha, data vector u, data int n){
      vector[n] x = col(xy, 1);
      vector[n] y = col(xy, 2);
      real v = (alpha[1]-1) * sum(log(u)) + (alpha[2]-1) * sum(log(x-u));
      v += (alpha[3]-1) * sum(log(y-u)) + (alpha[4]-1) * sum(log1m(x+y-u));
      v += -n * log_multi_beta(alpha);
      return v;
   }
}
data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;
   vector<lower=0, upper=1>[n] u;
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
}
model {
    alpha ~ gamma(a, b);
    xy ~ log_bivariate_beta(alpha, u, n);
}
