functions {
   real log_multi_beta(array[] real theta){
       return lgamma(theta[1]) + lgamma(theta[2]) + lgamma(theta[3]) + lgamma(theta[4]) - lgamma(sum(theta));
   }
   real integrand1(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i){
       real s = x_r[1];
       real t = x_r[2];
       real v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
       return v;
   }
   real integrand2(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i){
       real s = x_r[1];
       real t = x_r[2];
       real v = exp((theta[1]-1)*log(x) + (theta[2]-1)*log(s-x) + (theta[3]-1)*log(t-x) + (theta[4]-1)*log1m(s+t-x)); 
       return v;
    }
   real integrand3(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i){
       real s = x_r[1];
       real t = x_r[2];
       real v;
       if (s+t-1 > 0) {
          if (s < t) {
              if (xc>=0) {
                v = x^(theta[1]-1) * xc^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-t-xc)^(theta[4]-1);
              }
              else {
                v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * (-xc)^(theta[4]-1);
              }
          } else {
              if (xc>=0) {
                v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * xc^(theta[3]-1) * (1-s-xc)^(theta[4]-1);
              }
              else {
                v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * (-xc)^(theta[4]-1);
              }
          }
       } else {
          if (s < t) {
              if (xc>=0) {
                v = x^(theta[1]-1) * xc^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-t-xc)^(theta[4]-1);
              }
              else {
                v = (-xc)^(theta[1]-1) * (s+xc)^(theta[2]-1) * (t+xc)^(theta[3]-1) * (1-s-t-xc)^(theta[4]-1);
              }
          } else {
              if (xc>=0) {
                v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * xc^(theta[3]-1) * (1-s-xc)^(theta[4]-1);
              }
              else {
                v = (-xc)^(theta[1]-1) * (s+xc)^(theta[2]-1) * (t+xc)^(theta[3]-1) * (1-s-t-xc)^(theta[4]-1);
              }
          }
       }
       return v;
   }
   real integrand4(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i){
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
   real log_bivariate_beta_lpdf(data array[] real xy, array[] real alpha, real lb, real ub, array[] int x_i, 
                                data real tolerance, data int integrand){
       real v = 0;
       if (integrand==1) {
          v = log(integrate_1d(integrand1, lb, ub, alpha, xy, x_i, tolerance)) - log_multi_beta(alpha);
       } else if (integrand==2) {
          v = log(integrate_1d(integrand2, lb, ub, alpha, xy, x_i, tolerance)) - log_multi_beta(alpha);
       } else if (integrand==3) {
          v = log(integrate_1d(integrand3, lb, ub, alpha, xy, x_i, tolerance)) - log_multi_beta(alpha);
       } else {
          v = log(integrate_1d(integrand4, lb, ub, alpha, xy, x_i, tolerance)) - log_multi_beta(alpha);
       }
       return v;
   }
}
data {
   int<lower=0> n;
   array[n,2] real<lower=0, upper=1> xy;
   array[4] real<lower=0> alpha;
   real<lower=0> tolerance;
   int<lower=1,upper=4> integrand;
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
generated quantities {
    array[n] real log_density;
    for(i in 1:n){
        log_density[i] = log_bivariate_beta_lpdf(xy[i] | alpha, lb[i], ub[i], x_i, tolerance, integrand);
    }
}