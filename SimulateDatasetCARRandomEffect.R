library(dplyr)
library(ggplot2)
library(MASS)

source("autoMultiInferenceFunctions.R")

#Function to calculate multinomial probabilities

prob_multi <- function(X, beta, k){
  
  p <- NULL
  
  for(color in 1:k){
    
    c_beta <- beta[,color]
    
    temp <- exp(X%*%beta)
    
    p <- cbind(p, exp(X%*%c_beta)/apply(temp, 1, sum))
    
  }
  
  return(p)
  
}

#############################
##Create covariates##########
#############################

m1 <- 30
m2 <- 30

##Coords
c1 <- seq(from=0, to=1, length.out=m1)
c2 <- seq(from=0, to=1, length.out=m2)
n <- m1*m2

#x1 <- rep(c1, m1)
#x2 <- rep(c2, each=m2)

x1 <- rnorm(n, mean=2, sd=1)
x2 <- rnorm(n, mean=3, sd=1)

X <- cbind(rep(1, n), x1, x2)

p <- ncol(X)

##############################
##Set regression coefficients#
##############################

k <- 3

beta1 <- c(0,0,0) #Reference color
beta2 <- c(1, 0.5,-0.5)
beta3 <- c(-0.5, 1, -1)

beta <- cbind(beta1, beta2, beta3)

######################
##Set CAR parameters##
######################

nobj <- neigh_fun(m1, m2)
A <- neig_to_A(nobj)

alpha <- 0.5
tau <- 0.9
D <- diag(as.numeric(A%*%rep(1,n)))
W <- tau * (D - alpha * A)
Sigma <- solve(W)

#Simulate spatial effects

phi <- mvrnorm(n = 1, mu=rep(0, n), Sigma=Sigma)

####################
#Simulate response##
####################

mu <- X%*%beta + phi
probs <- prob_multi(X, beta, k)


#Simulate response
data <- sapply(1:n, function(i) which.max(rmultinom(1, size=1, prob=probs[i,])))

##Plot dataset

plot_df <- data.frame("x"=rep(c1, m1),
                      "y"=rep(c2, each=m2),
                      "value"=c(data))

plot_data <- ggplot(plot_df, aes(x=x, y=y, color=as.factor(value))) +
  geom_point(size=2.5)
plot_data

save(data, X, A, k, p, n, nobj, file="DatasetCAR.RData")
