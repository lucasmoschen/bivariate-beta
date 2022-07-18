// Bivariate beta model 
// Estimate the parameter U 
// It uses the transformation made in v5, but with alpha fixed.

data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;
   vector<lower=0>[4] alpha;
}
transformed data {
   vector[n] x = col(xy, 1);
   vector[n] y = col(xy, 2);
   vector[n] lb;
   vector[n] ub;
   for (i in 1:n) {
      lb[i] = max({1-x[i], 1-y[i]});
      ub[i] = min({2-x[i]-y[i], 1.0});
   }
}
parameters {
   vector<lower=lb, upper=ub>[n] z1;
}
model {
    target += log(z1) + (alpha[1]-1) * log(1-z1) + (alpha[2]-1) * log(z1+x-1);
    target += (alpha[4]-1) * log(2-(x+y+z1)) + (alpha[3]-1) * log(z1+y-1);
}
