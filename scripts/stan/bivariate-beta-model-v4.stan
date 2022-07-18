// Bivariate beta model 
// Estimate the parameter alpha 
// The variable U is  integrated through ODE (does not work)

functions {
    real log_multi_beta(vector theta){
        return lgamma(theta[1]) + lgamma(theta[2]) + lgamma(theta[3]) + lgamma(theta[4]) - lgamma(sum(theta));
    }
    vector ode(real t, vector y, vector theta, array[] real x_r) {
            real a = x_r[1];
            real b = x_r[2];
            vector[1] dydt;
            dydt[1] = t^(theta[1]-1) * (a-t)^(theta[2]-1) * (b-t)^(theta[3]-1) * (1-a-b+t)^(theta[4]-1);
            return dydt;
    }
    real log_bivariate_beta_lpdf(data array[] real xy, vector alpha, real lb, real ub, vector y0){
        array[1] vector[1] ode_solver = ode_rk45(ode, y0, lb+1e-8, {ub-1e-8}, alpha, xy);
        return log(ode_solver[1][1]) - log_multi_beta(alpha);
    }
}
data {
   int<lower=0> n;
   array[n,2] real<lower=0, upper=1> xy;
   array[4] real<lower=0> a;
   array[4] real<lower=0> b;
}
transformed data {
   array[n] real lb;
   array[n] real ub;
   for (i in 1:n) {
      lb[i] = max({0.0, xy[i,1]+xy[i,2]-1});
      ub[i] = min(xy[i]);
   }
   vector[1] y0 = rep_vector(0.0, 1);
}
parameters {
   vector<lower=0>[4] alpha;
}
model {
    alpha ~ gamma(a, b);
    for(i in 1:n){
        xy[i] ~ log_bivariate_beta(alpha, lb[i], ub[i], y0);
    }
}