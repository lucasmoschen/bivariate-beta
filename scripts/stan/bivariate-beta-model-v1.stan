// Bivariate beta model 
// Estimate the parameter alpha 
// The likelihood function depends on an integration

functions {
   real log_multi_beta(array[] real theta){
       return sum(lgamma(theta)) - lgamma(sum(theta));
   }
   real integrand(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i){
       real s = x_r[1];
       real t = x_r[2];
       real v;
       if (s+t-1 > 0){
           if (s < t){
               if (xc>=0){
                   v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(xc) + (theta[3]-1)*log(t-x) + (theta[4]-1)*log1m(xc+t));
               } else {
                   v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(s-x) + (theta[3]-1)*log(t-x) + (theta[4]-1)*log(-xc));
               }
           } else {
               if (xc>=0){
                   v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(s-x) + (theta[3]-1)*log(xc) + (theta[4]-1)*log1m(xc+s));
               } else {
                   v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(s-x) + (theta[3]-1)*log(t-x) + (theta[4]-1)*log(-xc));
               }
           }
       } else {
           if (s < t){
               if (xc>=0){
                   v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(xc) + (theta[3]-1)*log(t-x) + (theta[4]-1)*log1m(xc+t));
               } else {
                   v = exp((theta[1]-1)*log(-xc) + (theta[2]-1)*log(s+xc) + (theta[3]-1)*log(t+xc) + (theta[4]-1)*log1m(s+t+xc));
               }
           } else {
               if (xc>=0){
                   v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(s-x) + (theta[3]-1)*log(xc) + (theta[4]-1)*log1m(xc+s));
               } else {
                   v = exp((theta[1]-1)*log(-xc) + (theta[2]-1)*log(s+xc) + (theta[3]-1)*log(t+xc) + (theta[4]-1)*log1m(s+t+xc));
               }
           }
       }
       return v;
    }
    real log_bivariate_beta_lpdf(data array[] real xy, array[] real alpha, data real lb, data real ub, data array[] int x_i, data real tolerance){
       return log(integrate_1d(integrand, lb, ub, alpha, xy, x_i, tolerance));
    }
}
data {
   int<lower=0> n;
   array[n,2] real<lower=0, upper=1> xy;
   array[4] real<lower=0> a;
   array[4] real<lower=0> b;
   real<lower=0> tolerance;
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
}
model {
    target += gamma_lpdf(alpha | a, b);
    for(i in 1:n){
        target += log_bivariate_beta_lpdf(xy[i] | alpha, lb[i], ub[i], x_i, tolerance);
    }
    target += -n * log_multi_beta(alpha);
}