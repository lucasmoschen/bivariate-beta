// Bivariate beta model 
// Estimate the latent variable U
// The alpha parameter is fixed and for U is used the same model as v3.

data {
   int<lower=0> n;
   vector<lower=0, upper=1>[n] x;
   vector<lower=0, upper=1>[n] y;

   vector<lower=0>[4] a;
   vector<lower=0>[4] b;

}
transformed data {
   vector[n] lb;
   vector[n] ub;
   for(i in 1:n){
     lb[i] = max({0.0, x[i]+y[i]-1.0});
     ub[i] = min({x[i], y[i]});
   }
}
parameters {
   vector<lower=0>[4] alpha;
   vector<lower=lb, upper=ub>[n] u;
}
transformed parameters {
   array[n] simplex[4] theta;
   for(i in 1:n){
     theta[i,1] = u[i];
     theta[i,2] = x[i]-u[i];
     theta[i,3] = y[i]-u[i];
     theta[i,4] = 1-x[i]-y[i]+u[i];
   }
}
model {
   alpha ~ gamma(a,b);
   u ~ uniform(lb, ub);
   theta ~ dirichlet(alpha);
}
