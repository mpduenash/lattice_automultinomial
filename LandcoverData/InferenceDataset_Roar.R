library(automultinomial)

load("ProcessedReducedDataset3600.RData")
source("autoMultiInferenceFunctionsImproved.R")

outer <- 20
inner <- 4

###########################
###Double MH###############
###########################

#####Initial values for parameters: pseudolikelihood

yfac <- as.factor(data)
X <- X_full
p <- ncol(X)

pseudo_est <- MPLE(X=X, y=yfac, A=A, ciLevel = 0.95)
initial_beta <- pseudo_est$betaHat
initial_gamma <- pseudo_est$gammaHat

# 
# predpseudo <- drawSamples(initial_beta,initial_gamma,
#                             X,A,nSamples = 1, burnIn = 1000)
# 
# pred_df <- data.frame("x"=coords[,1],
#                       "y"=coords[,2],
#                       "value"=(predpseudo))
#  
# plot_pred <- ggplot(pred_df, aes(x=x, y=y, color=as.factor(value))) +
#   geom_point() +
#   ggtitle("Pseudolikelihood")
# plot_pred
#  
#####Parameters for DMH

#Number of chains
nchains <- 1

sigma_gamma <- 0.01
sigma_beta <- 0.0003

star <- proc.time()
sample <- autoMultiDMH(data=data, X=X, k=k, p=p, nobj=nobj, 
                         outer=outer, inner=inner, 
                         initial_gamma=initial_gamma, 
                         initial_beta=initial_beta, 
                         sigma_gamma=sigma_gamma, sigma_beta=sigma_beta)
end <- proc.time()-star

save(sample, file="FullPostSampleInner4Elevation.RData")





