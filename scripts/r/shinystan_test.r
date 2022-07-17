library("rstan")
library("shinystan")
library("dplyr")

options(mc.cores = parallel::detectCores())

true_alpha <- c(2.5,0.1,1.9,1.7)
n <- 50

Z1 = rbeta(n=n, sum(true_alpha[2:4]), true_alpha[1])
Z2 = rbeta(n=n, sum(true_alpha[3:4]), true_alpha[2])
Z3 = rbeta(n=n, true_alpha[4], true_alpha[3])
U = 1 - Z1
X = 1 - Z1*Z2
Y = 1 - Z1 + Z1*Z2*(1-Z3)
XY <- cbind(X,Y)

a <- c(1,1,1,1)
b <- c(1,1,1,1)

data <- list(n = n,
             u = U,
             xy = XY,
             a = a,
             b = b,
             tolerance=1e-8)

mod <- stan_model(file="Documents/GitHub/bivariate-beta/scripts/stan/bivariate-beta-model-v3.stan")
fit <- sampling(mod, data=data, warmup=1000, iter=2000)

# Optional visualizations

## Shinystan visualization
shiny_obj <- as.shinystan(fit)
launch_shinystan(shiny_obj)

## MCMC mean of alpha2
#f <- extract(fit, 'alpha', permuted=FALSE, inc_warmup=FALSE)
#alpha2_samples <- f[,1,2] #first chain
#plot(cummean(alpha2_samples), ylim=c(0.05, 0.5), ylab="MCMC mean", xlab="Iteration")
#title("Comparing the true alpha2 value and MCMC mean")
#abline(h=true_alpha[2], col='blue')
