library("rstan")
library("shinystan")
library("dplyr")

options(mc.cores = parallel::detectCores())

true_alpha <- c(2.5, 0.01, 1.9, 1.7)
n <- 1
U1 <- rbeta(n=n, shape1=2.5, shape2=0.1)
U2 <- rbeta(n=1, shape1=1.9, shape2=1.7)

data <- list(x = U1 + U2,
             a = 2.5,
             b = 0.1,
             c = 1.9,
             d = 1.7)

mod <- stan_model(file="Documents/GitHub/bivariate-beta/scripts/stan/testing.stan")
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
