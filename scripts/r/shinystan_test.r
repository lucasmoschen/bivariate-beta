library("rstan")
library("shinystan")

true_alpha <- c(2.5,0.1,1.9,1.7)
n <- 50
U1 <- rgamma(n=n, shape=true_alpha[1], rate=1)
U2 <- rgamma(n=n, shape=true_alpha[2], rate=1)
U3 <- rgamma(n=n, shape=true_alpha[3], rate=1)
U4 <- rgamma(n=n, shape=true_alpha[4], rate=1)
X <- (U1 + U2)/(U1 + U2 + U3  + U4)
Y <- (U1 + U3)/(U1 + U2 + U3  + U4)
XY <- cbind(X,Y)

a <- c(1,1,1,1)
b <- c(1,1,1,1)

data <- list(n = n,
             xy = XY,
             a = a,
             b = b)

fit <- stan(file="Documents/GitHub/bivariate-beta/scripts/stan/bivariate-beta-model-v3.stan",
            data=data)
shiny_obj <- as.shinystan(fit)
launch_shinystan(shiny_obj)
