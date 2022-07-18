// Bivariate beta model 
// Estimate the latent variable U
// The alpha parameter is fixed and for U is used the same model as v3.

data {
   real<lower=0, upper=1> x;
   real<lower=0, upper=1> y;
   vector<lower=0>[4] alpha;
}
transformed data {
   real lb = max({0.0, x+y-1});
   real ub = min({x, y});
}
parameters {
   real<lower=lb, upper=ub> u;
}
transformed parameters {
   simplex[4] theta;
   theta[1] = u;
   theta[2] = x-u;
   theta[3] = y-u;
   theta[4] = 1-x-y+u;
}
model {
    u ~ uniform(lb, ub);
    theta ~ dirichlet(alpha);
}
