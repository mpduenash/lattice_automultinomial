library(automultinomial)

#Load dataset
load("ProcessedReducedDataset3600.RData")
source("autoMultiInferenceFunctionsImproved.R")

#####Initial values for parameters: pseudolikelihood

yfac <- as.factor(data)
X <- X_full
p <- ncol(X)

pseudo_est <- MPLE(X=X, y=yfac, A=A, ciLevel = 0.95)
initial_beta <- pseudo_est$betaHat
initial_gamma <- pseudo_est$gammaHat

############################
###Define parameter blocks##
############################

ini_val <- matrix(c(initial_beta, initial_gamma), nrow=1)

d1 <- 5
d2 <- 5

ib <- list()
ib[[1]] <- 1:d1
ib[[2]] <- (d1+1):(d1+d2)
ib[[3]] <- d1+d2+1

#############################
####Set proposal covariance##
#############################

#scale constant
sc <- 2.1^2/d1

#Estimated variance from pseudolikelihood
Sigma_plk <- pseudo_est$variance

#Correction to make it positive definite
make_pd_eig <- function(Sigma, eps = 1e-8) {
  Sigma <- 0.5*(Sigma + t(Sigma))           # symmetrize
  ev <- eigen(Sigma, symmetric = TRUE)
  vals <- pmax(ev$values, eps)
  Sigma_pd <- ev$vectors %*% diag(vals) %*% t(ev$vectors)
  return(0.5*(Sigma_pd + t(Sigma_pd)))
}
# positive definite matrix
Sigma_fixed <- make_pd_eig(Sigma_plk, eps = 1e-8)

#Cholesky decomposition

Sigma_fixed <- chol(Sigma_fixed)

#Covariance for proposal at each block
cov_p <- list()

cov_p[[1]] <- sc*Sigma_fixed[1:d1,1:d1]
cov_p[[2]] <- sc*Sigma_fixed[(d1+1):(d1+d2),(d1+1):(d1+d2)]
cov_p[[3]] <- 2.38^2*Sigma_fixed[d1+d2+1,d1+d2+1]


########################################
####Double Metropolis Hastings by block#
########################################

#Number of chains
nchains <- 1
#Number of blocks
nb <- 3

#Outer samples
outer <- 100
inner <- 5

star <- proc.time()
result <- autoMultiDMH_block(data=data, X=X, k=k, p=p, nobj=nobj, 
                             outer=outer, inner=inner, ini_val=ini_val,
                             ib=ib, cov_p=cov_p)
end <- proc.time()-star
end

save(result, file="ResultBlockInner5.RData")