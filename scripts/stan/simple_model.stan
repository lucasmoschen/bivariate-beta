data {
   int<lower=0> n;
   vector<lower=0, upper=2>[n] x;
   
   real<lower=0> a;
   real<lower=0> b;
   real<lower=0> c;
   real<lower=0> d;
}
transformed data {
  vector[n] ub;
  vector[n] lb;
  for(i in 1:n){
    ub[i] = min({x[i], 1.0});
    lb[i] = max({x[i]-1.0, 0.0});
  }
}
parameters {
   vector<lower=lb, upper=ub>[n] u;
}
model {
    target += beta_lpdf(u | a,b);
    target += beta_lpdf(x-u | c,d);
}
