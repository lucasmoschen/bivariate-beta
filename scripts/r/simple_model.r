library("rstan")
library("shinystan")
library("dplyr")

options(mc.cores = parallel::detectCores())

true_alpha <- c(2.5, 0.1, 1.9, 1.7)
n <- 1
U1 <- rbeta(n=n, shape1=2.5, shape2=0.1)
U2 <- rbeta(n=1, shape1=1.9, shape2=1.7)

data <- list(x = U1 + U2,
             a = 2.5,
             b = 0.1,
             c = 1.9,
             d = 1.7)

mod <- stan_model(file="Documents/GitHub/bivariate-beta/scripts/stan/simple_model.stan")
fit <- sampling(mod, data=data, warmup=1000, iter=2000)

# Optional visualizations

## Shinystan visualization
shiny_obj <- as.shinystan(fit)
launch_shinystan(shiny_obj)
