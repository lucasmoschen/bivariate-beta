functions {
   real log_multi_beta(real[] theta){
       return lgamma(theta[1]) + lgamma(theta[2]) + lgamma(theta[3]) + lgamma(theta[4]) - lgamma(sum(theta));
   }
   real integrand(real x, real xc, real[] theta, real[] x_r, int[] x_i){
       real s = x_r[1];
       real t = x_r[2];
       return x^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
   }
   real log_bivariate_beta_lpdf(data real[] xy, real[] alpha, real lb, real ub, int[] x_i, data real tolerance){
       return log(integrate_1d(integrand, lb, ub, alpha, xy, x_i, tolerance)) - log_multi_beta(alpha);
   }
}
data {
   int<lower=0> n;
   real<lower=0, upper=1> xy[n,2];
   real<lower=0> a[4];
   real<lower=0> b[4];
   real<lower=0> tolerance;
}
transformed data {/* ... declarations ... statements ... */
   real lb[n];
   real ub[n];
   for (i in 1:n) {
      lb[i] = max({0.0, xy[i,1]+xy[i,2]-1})+1e-8;
      ub[i] = min(xy[i])-1e-8;
   }
   int x_i[0];
}
parameters {
   real<lower=0> alpha[4];
}
model {
    for(i in 1:4){
        alpha[i] ~ gamma(a[i], b[i]);
    }
    for(i in 1:n){
        xy[i] ~ log_bivariate_beta(alpha, lb[i], ub[i], x_i, tolerance);
    }
}

//    real integrand2(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i){
//        real s = x_r[1];
//        real t = x_r[2];
//        real values[4] = {x, s-x, t-x, 1-s-t+x};
//        int argmin = 1;
//        for (i in 2:4){
//            if (values[i] < values[argmin]) {
//               argmin = i;
//            }
//        }
//        values[argmin] = xc;
//        real v = 1.0;
//        for (i in 1:4){
//            v = v * values[i]^(theta[i]-1);
//        }
    //    if (s+t-1 > 0) {
    //       if (s < t) {
    //           if (x > s + (t-1)/2) {
    //             v = x^(theta[1]-1) * xc^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
    //           }
    //           else {
    //             v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * xc^(theta[4]-1);
    //           }
    //       } else {
    //           if (x > t + (s-1)/2) {
    //             v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * xc^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
    //           }
    //           else {
    //             v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * xc^(theta[4]-1);
    //           }
    //       }
    //    } else {
    //       if (s < t) {
    //           if (x > s/2) {
    //             v = x^(theta[1]-1) * xc^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
    //           }
    //           else {
    //             v = xc^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
    //           }
    //       } else {
    //           if (x > t/2) {
    //             v = x^(theta[1]-1) * (s-x)^(theta[2]-1) * xc^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
    //           }
    //           else {
    //             v = xc^(theta[1]-1) * (s-x)^(theta[2]-1) * (t-x)^(theta[3]-1) * (1-s-t+x)^(theta[4]-1);
    //           }
    //       }
    //    }
//        return v;
//    }