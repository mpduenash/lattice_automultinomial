library(ggplot2)
library(reticulate)
library(doParallel)
library(foreach)
library(automultinomial)

source("autoMultiInferenceFunctionsImproved.R")

#Load previously simulated dataset
load("DatasetCAR.RData")

#Send to HPC
args <- commandArgs(trailingOnly = TRUE)
task_id <- as.numeric(args[1])

#task_id <- 1

params <- read.csv("paramsCAR.csv", sep=";")
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

#Parameters for algorithm

nchains <- 1

sigma_gamma <- 0.01
sigma_beta <- 0.05

#Run

sample <- autoMultiDMH(data=data, X=X, k=k, p=p, nobj=nobj, 
                       outer=outer, inner=inner, 
                       initial_gamma=initial_gamma, 
                       initial_beta=initial_beta, 
                       sigma_gamma=sigma_gamma, sigma_beta=sigma_beta)

save(sample, file = sprintf("postSampleCAR_inner%d.RData", inner))


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

###############################
####Diagnostics################
###############################

#Auxiliary samples 
ini_cycle <- 1000

# Detect SLURM cores
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK", unset = 1))

# Setup parallel backend
cl <- makeCluster(n_cores)
registerDoParallel(cl)

##Start parallelize

dvalues <- foreach(i= 1:nsample, .combine = rbind) %dopar% {
  
  library(reticulate)
  library(automultinomial)
  
  numpy <- import("numpy")
  py_require("jax")
  jax <- import("jax")
  jnp <- import("jax.numpy")
  
  
  #Call functions
  source_python("FunctionACDAutomultinomial.py")
  
  
  beta_temp <- expand_beta(u_beta[i,], p, k)
  
  
  #Sample from auxiliary variables
  Y_aux <- drawSamples(beta=beta_temp,
                       gamma=u_gamma[i],
                       X=X,A=A,nSamples = n_aux, burnIn = ini_cycle)
  
  #Calculate d value for sample
  
  d_vec <- dFun(Y=data, X=X, Y_aux=t(Y_aux),
                beta=beta_temp, gamma=u_gamma[i], neig=A, sdBeta=1)
  
  d_vec
  
}

# Stop cluster
stopCluster(cl)

#Bind repeated dvalues
count_row_repeats_simple <- function(A, B) {
  apply(A, 1, function(r) sum(apply(B, 1, identical, r)))
}

#Bind repeated dvalues
nrep  <- count_row_repeats_simple(u_sample, thin_sample)
dfull <-  dvalues[rep(seq_len(nrow(dvalues)), nrep), ]

Sigma <- f_Vhat_bm(d=dfull, N=n_aux)
acd <- ACD_bm(d=dfull, N=n_aux)


save(acd, thin_sample, dfull, file = sprintf("dVecCAR_id%d.RData", task_id))