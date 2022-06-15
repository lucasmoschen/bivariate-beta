functions {
   real log_multi_beta(array[] real theta){
       return lgamma(theta[1]) + lgamma(theta[2]) + lgamma(theta[3]) + lgamma(theta[4]) - lgamma(sum(theta));
   }
   real log_bivariate_beta_lpdf(data array[] real xy, array[] real alpha, real u){
      real x = xy[1];
      real y = xy[2];
      // log(x-u) = log(x(1-u/x)) = log(x) + log(1-u/x) 
      real v = (alpha[2]-1) * (log(x)+log1m(u/x)) + (alpha[3]-1) *(log(y)+log1m(u/y)) + (alpha[4]-1) * log1m(x+y-u) - log_multi_beta(alpha);
      v += lbeta(alpha[1], alpha[2]+alpha[3]+alpha[4]) - (alpha[2]+alpha[3]+alpha[4]-1) * log1m(u);
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
   array[n,2] real<lower=0, upper=1> xy;
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
   array[n] real lb;
   array[n] real ub;
   for (i in 1:n) {
      lb[i] = max({0.0, xy[i,1]+xy[i,2]-1});
      ub[i] = min(xy[i]);
   }
   array[0] int x_i;
}
parameters {
   array[4] real<lower=0> alpha;
   array[n] real<lower=lb, upper=ub> u;
}
model {
   if (prior_alpha==1) {
      for (i in 1:4) {
       alpha[i] ~ gamma(a[i], b[i]);
      }
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
   for(i in 1:n){
      u[i] ~ beta(alpha[1], alpha[2]+alpha[3]+alpha[4]);
      xy[i] ~ log_bivariate_beta(alpha, u[i]);
   }
}