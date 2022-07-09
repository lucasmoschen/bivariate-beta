functions {
   real log_multi_beta(array[] real theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data matrix xy, array[] real alpha, vector u, int n){
      vector[n] x = col(xy, 1);
      vector[n] y = col(xy, 2);
      real v = (alpha[2]-1) * sum(log(x-u)) + (alpha[3]-1) * sum(log(y-u)) + (alpha[4]-1) * sum(log1m(x+y-u));
      v += -(alpha[2]+alpha[3]+alpha[4]-1) * sum(log1m(u));
      v += - n * log_multi_beta(alpha[2:4]);
      return v;
   }
   real log_boojum_lpdf(array[] real x, real m, array[] real r){
      return -m * log_multi_beta(x) - dot_product(r, x);
   }
   real log_gamma_00_lpdf(array[] real x){
      return -sum(log(x));
   }
   real log_gamma_10_lpdf(array[] real x){
      return 0;
   }
   real log_uniform_exponential_lpdf(real x, real c, real lambda, real p){
      if (x < c) {
         return log(p) - log(c);
      } else {
         return log(lambda) + log1m(p) - lambda*(x-c);
      }
   }
}
data {
   int<lower=0> n;
   matrix<lower=0, upper=1>[n,2] xy;
   int prior_alpha;
   // gamma prior
   array[4] real<lower=0> a;
   array[4] real<lower=0> b;

   // bojum prior
   real<lower=-1> m;
   array[4] real<lower=0> r;

   // uniform-exponential prior
   array[4] real<lower=0, upper=1> p;
   array[4] real<lower=0> lambda;
   array[4] real<lower=0> c;
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
   array[4] real<lower=0> alpha;
   vector<lower=lb, upper=ub>[n] u;
}
model {
   if (prior_alpha==1) {
      alpha ~ gamma(a, b);
   } else if (prior_alpha==2) {
      alpha ~ log_boojum(m, r);
   } else if (prior_alpha==3) {
      alpha ~ log_gamma_00(); //Gamma(0,0)
   } else if (prior_alpha==4) {
      for (i in 1:4) {
        alpha[i] ~ log_uniform_exponential(c[i], lambda[i], p[i]); 
      }
   } else {
      alpha ~ log_gamma_10(); //Gamma(1,0)
   }
   u ~ beta(alpha[1], alpha[2]+alpha[3]+alpha[4]);
   xy ~ log_bivariate_beta(alpha, u, n);
}