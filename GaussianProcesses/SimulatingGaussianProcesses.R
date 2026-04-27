library(mvtnorm)
library(ggplot2)
library(sf)
library(dplyr)
library(foreach)
library(doParallel)
library(automultinomial)

source("autoMultiInferenceFunctions.R")

###############
##Create grid##
###############

cell_n <- 30
tot <- cell_n*cell_n

x <- seq(from=1, to=cell_n, by=1)
y <- seq(from=1, to=cell_n, by=1)

points <- data.frame("x"=rep(x, cell_n),
                     "y"=rep(y, each=cell_n))

points <- st_as_sf(points, coords=c("x", "y"))

#Matrix of distances

centr_dist <- as.numeric(st_distance(points))
#matrix(centr_dist, nrow=tot, byrow = TRUE)

########################################################
#Matrix of covariance (Exponentiated quadratic kernel)##
########################################################

l <- 3
Sigma_X1 <-  exp(-centr_dist^2/(2*l^2))
Sigma_X1 <- matrix(Sigma_X1, nrow=tot, byrow = TRUE)

l <- 0
Sigma_X2 <-  exp(-centr_dist^2/(2*l^2))
Sigma_X2 <- matrix(Sigma_X2, nrow=tot, byrow = TRUE)

########################
###Simulate covariates##
########################

set.seed(3589)

X1 <- rnorm(tot, 0, 2)
# X2 <- as.numeric(rmvnorm(1, mean=rep(0, tot), sigma=Sigma_X2))
X2 <- rnorm(tot, 1, 3)

#X1 <- rep(1:cell_n, each=cell_n)
#X2 <- rep(cell_n:1, cell_n)

X <- cbind(X1, X2)

ncolor <- 3

beta1 <- c(0,0)
beta2 <- c(-0.5,1)
beta3 <- c(2,-1)

beta <- cbind(beta1, beta2, beta3)

meanGPs <-  X%*%beta

#######################
####Simulate GPs#######
#######################

set.seed(754)

l <- 3
Sigma_M <-  exp(-centr_dist^2/(2*l^2))
Sigma_M <- matrix(Sigma_M, nrow=tot, byrow = TRUE)

gp1 <- as.numeric(rmvnorm(1, mean=meanGPs[,1], sigma=Sigma_M))
gp2 <- as.numeric(rmvnorm(1, mean=meanGPs[,2], sigma=Sigma_M))
gp3 <- as.numeric(rmvnorm(1, mean=meanGPs[,3], sigma=Sigma_M))

###Plot the Gaussian Processes

#GP1

df_temp1 <- data.frame("x"=rep(1:cell_n, each=cell_n),
                       "y"=rep(cell_n:1, cell_n),
                       "value"=gp1)

map_gp1 <- ggplot(df_temp1, aes(x=x, y=y, color=value)) +
  geom_point(size=3.5)  +
  labs(color = "Color") +
  theme(
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.background = element_rect(fill='transparent'),
    legend.box.background = element_rect(fill='transparent')
  )
map_gp1


#GP2

df_temp2 <- data.frame("x"=rep(1:cell_n, each=cell_n),
                       "y"=rep(cell_n:1, cell_n),
                       "value"=gp2)

map_gp2 <- ggplot(df_temp2, aes(x=x, y=y, color=value)) +
  geom_point(size=3.5)  +
  labs(color = "Color") +
  theme(
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.background = element_rect(fill='transparent'),
    legend.box.background = element_rect(fill='transparent')
  )
map_gp2


#GP3

df_temp3 <- data.frame("x"=rep(1:cell_n, each=cell_n),
                       "y"=rep(cell_n:1, cell_n),
                       "value"=gp3)

map_gp3 <- ggplot(df_temp3, aes(x=x, y=y, color=value)) +
  geom_point(size=3.5)  +
  labs(color = "Color") +
  theme(
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.background = element_rect(fill='transparent'),
    legend.box.background = element_rect(fill='transparent')
  )
map_gp3

##############################
###Generate categorical data##
##############################

matrix_gp <- cbind(gp1, gp2, gp3)
final_gp <- apply(matrix_gp, 1, which.max)

#Plot simulated arrangement

df_tempX <- data.frame("x"=rep(1:cell_n, each=cell_n),
                       "y"=rep(cell_n:1, cell_n),
                       "value"=final_gp)

map_X <- ggplot(df_tempX, aes(x=x, y=y, color=as.factor(value))) +
  geom_point(size=3)
map_X

#########################
##Fit with DMH###########
#########################

#Observed data

df_tempY <- data.frame("x"=rep(1:cell_n, each=cell_n),
                       "y"=rep(cell_n:1, cell_n),
                       "value"=final_gp)

map_original <- ggplot(df_tempX, aes(x=x, y=y, color=as.factor(value))) +
  geom_point(size=3) +
  ggtitle("Observed arrangement")
map_original

Y <- final_gp

##Energy original data

nobj <- neigh_fun(cell_n, cell_n )
A <- neig_to_A(nobj)
data_energy <- energy(Y, nobj = nobj)

###Fitting with autologistic

yfac <- as.factor(Y)

pseudo_est <- MPLE(X=X, y=yfac, A=A, ciLevel = 0.95)

pseudo_beta <- pseudo_est$betaHat
pseudo_gamma <- pseudo_est$gammaHat
pseudo_probs <- pseudo_est$conditionalProbabilities

pred_pseudo <- drawSamples(beta=pseudo_beta,gamma=pseudo_gamma,
                          X=X, A=A, nSamples = 1, burnIn = 1000)

df_pred <- data.frame("x"=rep(1:cell_n, each=cell_n),
                      "y"=rep(cell_n:1, cell_n),
                      "value"=pred_pseudo)


plot_pred <- ggplot(df_pred, aes(x=x, y=y, color=as.factor(value))) +
  geom_point(size=3) +
  ggtitle("Prediction pseudo")

plot_pred


data <- Y
k <- ncolor
p <- 2
n <- tot

save(data, X, A, k, p, n, nobj, file="DatasetGP.RData")

