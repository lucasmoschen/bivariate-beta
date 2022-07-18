data {
   real<lower=0, upper=2> x;
   
   real<lower=0> a;
   real<lower=0> b;
   real<lower=0> c;
   real<lower=0> d;
}
transformed data {
  real ub = x<1.0 ? x : 1.0;
}
parameters {
   real<lower=0, upper=ub> u;
}
model {
    target += beta_lpdf(u | a,b);
    target += beta_lpdf(x-u | c,d);
}
