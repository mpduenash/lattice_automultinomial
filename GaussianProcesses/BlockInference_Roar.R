library(reticulate)
library(doParallel)
library(foreach)
library(automultinomial)
library(ggplot2)

source("autoMultiInferenceFunctions.R")

#Load previously simulated dataset
load("DatasetGP.RData")
p <- 2

#Send to HPC
args <- commandArgs(trailingOnly = TRUE)
task_id <- as.numeric(args[1])

#task_id <- 1

params <- read.csv("paramsGP.csv", sep=";")
outer <- params$outer[task_id]
inner <- params$inner[task_id]
n_aux <- params$n_aux[task_id]
burnin <- params$burnin[task_id]
thin <- params$thin[task_id]

###########################
###Double MH###############
###########################

m1 <- 30
m2 <- 30

##Coords
c1 <- seq(from=0, to=1, length.out=m1)
c2 <- seq(from=0, to=1, length.out=m2)

#####Initial values for parameters: pseudolikelihood

yfac <- as.factor(data)

pseudo_est <- MPLE(X=X, y=yfac, A=A, ciLevel = 0.95)
initial_beta <- pseudo_est$betaHat
initial_gamma <- pseudo_est$gammaHat

############################
###Define parameter blocks##
############################

ini_val <- matrix(c(initial_beta, initial_gamma), nrow=1)

d1 <- 2
d2 <- 2

ib <- list()
ib[[1]] <- 1:d1
ib[[2]] <- (d1+1):(d1+d2)
ib[[3]] <- d1+d2+1

#############################
####Set proposal covariance##
#############################

#scale constant
sc <- 1.7^2/d1

#Estimated variance from pseudolikelihood
Sigma_plk <- pseudo_est$variance

#Cholesky decomposition

Sigma_fixed <- chol(Sigma_plk)

#Covariance for proposal at each block
cov_p <- list()

cov_p[[1]] <- sc*Sigma_fixed[1:d1,1:d1]
cov_p[[2]] <- sc*Sigma_fixed[(d1+1):(d1+d2),(d1+1):(d1+d2)]
cov_p[[3]] <- 1.7^2*Sigma_fixed[d1+d2+1,d1+d2+1]

########################################
####Double Metropolis Hastings by block#
########################################

#Number of chains
nchains <- 1
#Number of blocks
nb <- 3

# #Outer samples
# outer <- 1000
# inner <- 5

star <- proc.time()
result <- autoMultiDMH_block(data=data, X=X, k=k, p=p, nobj=nobj, 
                             outer=outer, inner=inner, ini_val=ini_val,
                             ib=ib, cov_p=cov_p)
end <- proc.time()-star
end

save(result, file = sprintf("postSampleGP_inner%d.RData", inner))

###########################
###Diagnostic##############
###########################

sample <- result$post_sample

sample_nburn <- sample[-c(1:burnin),]

##Thinnig chains

tot <- (outer*nchains-nchains*burnin)/thin

thin_sample <- sample_nburn[(1:tot)*thin,]


#Unique values 

u_sample <- unique(thin_sample)

u_beta <- u_sample[,c(1:(p*(k-1)))] 
u_gamma <- u_sample[,-c(1:(p*(k-1)))] 

#number of unique posterior values
nsample <- nrow(u_beta)

#Auxiliary samples 
ini_cycle <- 1000

# Detect SLURM cores
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK", unset = 1))
M
registerDoParallel(cores = n_cores)

##Start parallelize

dvalues <- foreach(i= 1:nsample, .combine = rbind) %dopar% {
  
  if (!exists("jax_initialized")) {
    library(reticulate)
    jax <- import("jax")
    source_python("FunctionACDAutomultinomial.py")
    jax_initialized <- TRUE
  }
  
  beta_temp <- expand_beta(u_beta[i,], p, k)
  
  
  #Sample from auxiliary variables
  Y_aux <- drawSamples(beta=beta_temp,
                       gamma=u_gamma[i],
                       X=X,A=A,nSamples = n_aux, burnIn = ini_cycle)
  
  #Calculate d value for sample
  
  d_vec <- dFun(Y=data, X=X, Y_aux=t(Y_aux),
                beta=beta_temp, gamma=u_gamma[i], neig=A, sdBeta=1)
  

  system("free -h | grep Mem")
  

  d_vec
  
}

stopImplicitCluster()

count_row_repeats_simple <- function(A, B) {
  apply(A, 1, function(r) sum(apply(B, 1, identical, r)))
}

#Bind repeated dvalues
nrep  <- count_row_repeats_simple(u_sample, thin_sample)
dfull <-  dvalues[rep(seq_len(nrow(dvalues)), nrep), ]

Sigma <- f_Vhat_bm(d=dfull, N=n_aux)
acd <- ACD_bm(d=dfull, N=n_aux)

save(acd, thin_sample, dfull, result, file = sprintf("dVecGP_inner%d.RData", inner))
