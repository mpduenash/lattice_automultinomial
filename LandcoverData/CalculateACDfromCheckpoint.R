library(reticulate)
library(doParallel)
library(foreach)
library(automultinomial)
library(Matrix)

####################################
#Load posterior samples for tasks###
####################################

load(paste0("checkpoint_iter_", 53000, ".RData"))

#Load dataset
load("ProcessedReducedDataset3600.RData")

#load functions
source("autoMultiInferenceFunctionsImproved.R")

####################################
####Prepare for diagnostic##########
####################################

#Batch 1

thin <- 5

post_sample <- post_sample[20000:53000,]
thin_sample <- post_sample[seq(from=1, to=nrow(post_sample), by=thin),]

u_sample <- unique(thin_sample)

u_beta <- u_sample[,c(1:(p*(k-1)))] 
u_gamma <- u_sample[,-c(1:(p*(k-1)))] 

nsample <- nrow(u_sample)

#Bind repeated dvalues
load("dvaluescheckpoint_iter_25000.RData")

d1 <- dvalues

load("dvaluescheckpoint_iter_30000.RData")

d2 <- dvalues

load("dvaluescheckpoint_iter_35000.RData")

d3 <- dvalues

load("dvaluescheckpoint_iter_53000.RData")

d4 <- dvalues

dfull <- rbind(d1, d2, d3, d4)

#Bind repeated values
# nparam <- p*(k-1)+1
# 
# nrep <- rep(0, nsample)
# 
# for(i in 1:nrow(u_sample)){
#   
#   for(j in 1:nrow(thin_sample)){
#     
#     nrep[i] <- nrep[i]+ (sum(u_sample[i,]==thin_sample[j,])==nparam)
#     
#   }
# }
# 
# nrep  <- count_row_repeats_simple(u_sample, thin_sample)
# dfull <-  dfull[rep(seq_len(nrow(dfull)), nrep), ]

n_aux <- 1000

Sigma <- f_Vhat_bm(d=dfull, N=n_aux)
acd <- ACD_bm(d=dfull, N=n_aux)

qchisq(0.99, rankMatrix(Sigma))

n <- nrow(dfull)

d_thinned <- dfull[seq(from=1, to=n, by=60),]
new_sample <- thin_sample[seq(from=1, to=n, by=60),]

Sigma <- f_Vhat_bm(d=d_thinned, N=n_aux)
acd2 <- ACD_bm(d=d_thinned, N=n_aux)
acd2

qchisq(0.99, rankMatrix(Sigma))


#############Check changes in posterior samples obtained from thining the sample

#Original posterior sample, thinned by 5 for computational advantage

df_density <- data.frame(thin_sample, "type"=rep("Original", nrow(thin_sample))) %>% 
  rbind(data.frame(new_sample, "type"=rep("Thinned", nrow(new_sample))))

density_p1 <- ggplot(df_density, aes(x=X11, color=type, fill=type)) +
  geom_density(alpha=0.3) +
  geom_vline(xintercept = mean(thin_sample[,11]),  linetype=5) +
  geom_vline(xintercept = mean(new_sample[,11]), linetype=3) +
  ggtitle("Parameter 11")
density_p1


###Try again

n_aux <- 1000

Sigma <- f_Vhat_bm(d=dfull, N=n_aux)
acd <- ACD_bm(d=dfull, N=n_aux)

qchisq(0.99, rankMatrix(Sigma))

n <- nrow(dfull)

d_thinned <- dfull[seq(from=1, to=n, by=15),]
new_sample <- thin_sample[seq(from=1, to=n, by=15),]

Sigma <- f_Vhat_bm(d=d_thinned, N=n_aux)
acd2 <- ACD_bm(d=d_thinned, N=n_aux)
acd2

qchisq(0.99, rankMatrix(Sigma))

#Original posterior sample, thinned by 5 for computational advantage

df_density <- data.frame(thin_sample, "type"=rep("Original", nrow(thin_sample))) %>% 
  rbind(data.frame(new_sample, "type"=rep("Thinned", nrow(new_sample))))

density_p1 <- ggplot(df_density, aes(x=X11, color=type, fill=type)) +
  geom_density(alpha=0.3) +
  geom_vline(xintercept = mean(thin_sample[,11]),  linetype=5) +
  geom_vline(xintercept = mean(new_sample[,11]), linetype=3) +
  ggtitle("Parameter 11")
density_p1

#####

tot <- matrix(0, nrow=ncol(dfull), ncol=ncol(dfull))

for(i in 1:nrow(dfull)){
  
 tot  <- tot + dfull[i,]%*%t(dfull[i,])
  
}

Vn <- 1/nrow(dfull)*tot
solve(Vn)
rankMatrix(Vn)
