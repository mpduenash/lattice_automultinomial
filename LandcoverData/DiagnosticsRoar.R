library(reticulate)
library(doParallel)
library(foreach)
library(automultinomial)

####################################
#Load posterior samples for tasks###
####################################

#Posterior distribution
load(paste0("checkpoint_iter_", 53000, ".RData"))

#Load dataset
load("ProcessedReducedDataset3600.RData")

#load functions
source("autoMultiInferenceFunctionsImproved.R")

####################################
####Prepare for diagnostic##########
####################################

thin <- 5

post_sample <- post_sample[35001:53000,]
thin_sample <- post_sample[seq(from=1, to=nrow(post_sample), by=thin),]

u_sample <- unique(thin_sample)

u_beta <- u_sample[,c(1:(p*(k-1)))] 
u_gamma <- u_sample[,-c(1:(p*(k-1)))] 

nsample <- nrow(u_sample)
X <- X_full

#Auxiliary samples 
ini_cycle <- 1000
n_aux <- 1000

# Detect SLURM cores
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK", unset = 1))

#n_cores <- 4

##Start

start <- Sys.time()
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
  
  
  beta_temp <- matrix(u_beta[i,], nrow=p, ncol=k-1)
  
 
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
stop <- Sys.time()-start
stop

# Save results per SLURM array job
save(dvalues, file =paste0("dvaluescheckpoint_iter_", 53000,".RData"))