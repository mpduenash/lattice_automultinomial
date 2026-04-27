library(automultinomial)

####Neighborhood function

neigh_fun <- function(n, m){
  
  lneigh <- list()
  
  for(i in 1:n){
    
    for(j in 1:m){
      
      sel <- i+(j-1)*n
      
      n1 <- ifelse((i-1)>=1,  sel-1, NA)
      n2 <- ifelse((i+1)<=n, sel+1, NA)
      n3 <- ifelse((j-1)>=1, sel-n, NA)
      n4 <- ifelse((j+1)<=m,  sel+n, NA)
      
      lneigh[[sel]] <- as.numeric(na.omit(c(n1,n2,n3,n4)))
      
    }
    
  }
  
  return(lneigh)
  
}

######Neighborhood to adjacency matrix

neig_to_A <- function(nobj){
  
  n <- length(nobj)
  A <- matrix(0, nrow=n, ncol=n)
  
  for(i in 1:n){
    A[i, nobj[[i]]] <- 1 
  }
  
  return(A)
}

######Energy function

energy <- function(X, nobj){
  
  X <- c(X)
  m <- length(X)-1
  tot <- 0
  
  for(i in 1:m){
    index <- nobj[[i]][nobj[[i]]>i]
    tot <- tot+sum(X[index]==X[i])
  }
  
  return(tot)
}

####Expand beta matrix

expand_beta <- function(beta_vec, p, k) {
  if (k == 1L) return(matrix(0, nrow = p, ncol = 1))
  mat <- matrix(beta_vec, nrow = p, ncol = k - 1, byrow = FALSE)
  mat
}

##############################
##Double Metropolis Hastings##
##############################

autoMultiDMH <- function(data, X, k, p, nobj, outer, inner, initial_gamma, 
                             initial_beta, sigma_gamma, sigma_beta){
  
  A <- neig_to_A(nobj)
  
  # new_gamma <- numeric(outer)
  # new_beta  <- matrix(NA, nrow = outer, ncol = p * (k - 1))
  
  new_gamma <- initial_gamma
  new_beta <- matrix(c(initial_beta), ncol=p * (k - 1), nrow=1)
  
  Sz <- energy(data, nobj)
  data_fac <- as.factor(data)
  
  for(i in 2:outer){
    
    prop_gamma <- new_gamma[i - 1] + rnorm(1, 0, sigma_gamma)
    prop_beta  <- expand_beta(new_beta[i - 1, ] + rnorm(p * (k - 1), 0, sigma_beta), p, k)
    
    if(prop_gamma>2 || prop_gamma<0.001){
      
      logprob <- -Inf
      
    } else{
      
      #Inner sampler
      Y <- drawSamples(beta=prop_beta, gamma=prop_gamma,
                       X=X, A=A, nSamples = 1, burnIn = inner, 
                       y=data_fac)
      
      #Compute acceptance probability
      D <- sapply(2:k, function(k) as.integer(c(Y) == k) - as.integer(data == k))
      
      curr_beta <- expand_beta(new_beta[i - 1, ], p, k)
      DeltaB <- curr_beta-prop_beta
      
      # find rows with any nonzero differences
      rows_use <- which(rowSums(abs(D)) != 0)
      if (length(rows_use) == 0) {
        score_beta_sum <- 0
      } else {
        V <- t(X[rows_use, , drop=FALSE]) %*% D[rows_use, , drop=FALSE]
        score_beta_sum <- sum(V * DeltaB)
      }
      
      Sy <- energy(Y, nobj)
      score_gamma <- (new_gamma[i - 1] - prop_gamma) * (Sy - Sz)
      
      logprob <- score_gamma+score_beta_sum
      
    }
    
    if(log(runif(1)) < logprob){
      new_gamma <- c(new_gamma, prop_gamma)
      new_beta <- rbind(new_beta, c(prop_beta))
      
    } else{
      new_gamma <- c(new_gamma, new_gamma[i-1])
      new_beta <- rbind(new_beta,new_beta[i-1,])
    }
    
    
  }
  
  return(cbind(new_beta, new_gamma))
  
}

########################Block update

autoMultiDMH_block <- function(data, X, k, p, nobj, outer, inner, ini_val,
                               ib, cov_p){
  
  A <- neig_to_A(nobj)
  
  #Data energy
  Sz <- energy(data, nobj)
  data_fac <- as.factor(data)
  
  post_sample <- ini_val
  tot_p <- ncol(post_sample)
  
  #Calculate acceptance probability
  
  ap <- rep(0, nb)
  
  #Starts outer loop 
  for(j in 2:outer){
    
    #Starts loop for blocks
    post_sample <- rbind(post_sample, post_sample[j-1,])
    
    for(i in 1:nb){
      
      prop_b <- post_sample[j-1,ib[[i]]]+ t(cov_p[[i]]) %*% rnorm(length(ib[[i]]))
      theta_prop <- post_sample[j-1,]
      theta_prop[ib[[i]]] <- prop_b
      
      new_beta <- expand_beta(theta_prop[-tot_p], p, k)
      new_gamma <- theta_prop[tot_p]
      
      #Inner sampler
      Y <- drawSamples(beta=new_beta, 
                       gamma=new_gamma,
                       X=X, A=A, nSamples = 1, burnIn = inner, 
                       y=data_fac)
      
      #Calculate acceptance probability according to block
      if(i!=nb){
        #Changes in color: from data to auxiliary sample
        D <- sapply(2:k, function(k) as.integer(c(Y) == k) - as.integer(data == k))
        
        curr_beta <- expand_beta(post_sample[j-1,-tot_p], p, k)
        DeltaB <- curr_beta-new_beta
        
        # find rows with any nonzero differences
        rows_use <- which(rowSums(abs(D)) != 0)
        if (length(rows_use) == 0) {
          score_beta_sum <- 0
        } else {
          V <- t(X[rows_use, , drop=FALSE]) %*% D[rows_use, , drop=FALSE]
          score_beta_sum <- sum(V * DeltaB)
        }
        
        #log of acceptance probability
        logprob <- score_beta_sum
        
      } else {
        
        #Change in energy
        Sy <- energy(Y, nobj)
        score_gamma <- (post_sample[j-1,ib[[i]]] - new_gamma) * (Sy - Sz)
        
        logprob <- score_gamma
        
      }
      
      #Accept or reject
      
      if(log(runif(1)) < logprob){
        
        post_sample[j,ib[[i]]] <- prop_b
        ap[i] <- ap[i]+1
        
      } else{
        post_sample[j,] <- post_sample[j,]
      }
      
    } #Close block loop
    
    if (j %% 10000 == 0) {
      save(result=post_sample,
           ap=ap,
           iter = j,
           file = paste0("checkpoint_iter_", j, ".RData"))
    }
    
  } #Close outer loop
  
  return(list("post_sample"=post_sample, "ap"=ap/outer))
  
} #Close function

###############################
#####Diagnostics functions#####
###############################

f_Vhat_bm = function(d, N){
  n = nrow(d)
  b = floor(min(n^(1/3), N^(1/3)))
  a = floor(n/b)
  
  dbarbk = sapply(1:a, function(k) return(colMeans(d[((k - 1) * b + 1):(k * b),])))
  dbar = rowMeans(dbarbk)
  dummy = 0
  for(k in 1:a){
    dummy = dummy + (dbarbk[,k] - dbar) %*% t(dbarbk[,k] - dbar)
  }
  Sigmahat = b * dummy / (a-1)
  
  return( Sigmahat )
}

ACD_bm = function(d, N){
  n = nrow(d)
  dbar = colMeans(d)
  res = n * t(dbar) %*% solve(f_Vhat_bm(d, N)) %*% dbar
  return(as.vector(res))
}