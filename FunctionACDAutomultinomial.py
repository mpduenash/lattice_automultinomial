#library(reticulate) #This is R code
#py_require("jax")
#py_require("numpy")

import jax
import jax.numpy as jnp
import numpy

#Create functions

def create_neighborhood_matrix_2d(n,m):
    rows = int(n)
    cols = int(m)
    num_nodes = rows * cols
    adj_matrix = numpy.zeros((num_nodes, num_nodes), dtype=int)

    for r in range(rows):
        for c in range(cols):
            current_node_idx = r * cols + c

            # Check neighbors: up, down, left, right
            neighbors = []
            if r > 0:  # Up
                neighbors.append((r - 1, c))
            if r < rows - 1:  # Down
                neighbors.append((r + 1, c))
            if c > 0:  # Left
                neighbors.append((r, c - 1))
            if c < cols - 1:  # Right
                neighbors.append((r, c + 1))

            for nr, nc in neighbors:
                neighbor_node_idx = nr * cols + nc
                adj_matrix[current_node_idx, neighbor_node_idx] = 1
                adj_matrix[neighbor_node_idx, current_node_idx] = 1  # Symmetric

    return adj_matrix

#Neighborhood matrix

#neig = create_neighborhood_matrix_2d((n,m))

###############################
#Functions for the auto model##
###############################

###1. Gradient of log of H

def logHAuto(Y, X, beta, gamma, neig):
    Y = jnp.asarray(Y).astype(jnp.int32).reshape(-1)
    X = jnp.asarray(X)
    beta = jnp.asarray(beta)
    neig = jnp.asarray(neig).astype(bool)
    p = beta.shape[1] + 1

    betaFull = jnp.column_stack((jnp.zeros((beta.shape[0], 1)), beta))
    betaY = betaFull[:, Y - 1].T   # shape (400, 2)

    linear_term = jnp.sum(X * betaY, axis=1)  # shape (400,)

    YiYj = (Y[:, None] == Y[None, :])
    energy = 0.5 * jnp.sum(neig * YiYj)

    logH = jnp.sum(linear_term) + gamma * energy
    return logH

def RlogHAuto(Y, X, beta, gamma, neig):
    result = logHAuto(Y, X, beta, gamma, neig)
    return numpy.array(result)

# Define gradient function ONCE and JIT it
grad_logHAuto = jax.jit(
    jax.grad(lambda b, g, Y, X, neig: logHAuto(Y, X, b, g, neig), argnums=(0, 1))
)

@jax.jit
def gradLogH(Y, X, beta, gamma, neig):
    # Compute gradients efficiently
    grad_beta, grad_gamma = grad_logHAuto(beta, gamma, Y, X, neig)
    return jnp.concatenate([grad_beta.ravel(), jnp.array([grad_gamma])])

  
def RgradLogH(Y, X, beta, gamma, neig):
    result = gradLogH(Y, X, beta, gamma, neig)
    return numpy.array(result)
  

###2. Gradient of log of prior

@jax.jit
def gradLogPrior(beta, gamma, sdBeta):
    grad_beta = (-1.0 / sdBeta) * beta
    grad_gamma = 0.0
    return jnp.concatenate([grad_beta.ravel(), jnp.array([grad_gamma])])


def RgradLogPrior(beta, gamma, sdBeta):
    result = gradLogPrior(beta, gamma, sdBeta)
    return numpy.array(result)

###3. Score function of posterior

def scorePosterior(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    T1 = gradLogH(Y, X, beta, gamma, neig)
    gradLogH_vmapped = jax.vmap(lambda y: gradLogH(y, X, beta, gamma, neig))
    grads_aux = gradLogH_vmapped(Y_aux)
    T2 = jnp.mean(grads_aux, axis=0)
    T3 = gradLogPrior(beta, gamma, sdBeta)
    Tot = T1 - T2 + T3
    return Tot

def RscorePosterior(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    result = scorePosterior(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    return numpy.array(result)
  
###4. J element of d

@jax.jit
def Jd(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    val = scorePosterior(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    A = jnp.outer(val, val)
    result = A[jnp.tril_indices(A.shape[0])]
    return result


def RJd(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    result = Jd(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    return numpy.array(result)

##5. Getting Hessians of logHAuto

@jax.jit
def hessianLogH(Y, X, beta, gamma, neig):
    p, k_minus1 = beta.shape
    theta = jnp.concatenate([beta.ravel(), jnp.array([gamma])])
    def wrapped_logHAuto(theta):
        beta_mat = theta[:-1].reshape(p, k_minus1)
        gamma_val = theta[-1]
        return logHAuto(Y, X, beta_mat, gamma_val, neig)
    Htheta = jax.hessian(wrapped_logHAuto)(theta)
    return Htheta


def RhessianLogH(Y, X, beta, gamma, neig):
    result = hessianLogH(Y, X, beta, gamma, neig)
    return numpy.array(result)


###6. Getting Hessian of logPrior

@jax.jit
def hessianLogPrior(beta, gamma, sdBeta):
    p, k_minus1 = beta.shape
    dim = p * k_minus1 + 1 
    diag_beta = jnp.full(p * k_minus1, -1.0 / sdBeta)
    diag_gamma = jnp.array([0.0])
    Htheta = jnp.diag(jnp.concatenate([diag_beta, diag_gamma]))
    return Htheta
  
def RhessianLogPrior(beta, gamma, sdBeta):
    result = hessianLogPrior(beta, gamma, sdBeta)
    return numpy.array(result)
    
###7. Approximation of hessian of intractable normalizing function

# Assuming hessianLogH and gradLogH are already JIT-compiled
hessianLogH_vmap = jax.vmap(lambda y, X, beta, gamma, neig: hessianLogH(y, X, beta, gamma, neig),
                            in_axes=(0, None, None, None, None))
gradLogH_vmap = jax.vmap(lambda y, X, beta, gamma, neig: gradLogH(y, X, beta, gamma, neig),
                         in_axes=(0, None, None, None, None))
                         
@jax.jit
def appHessc(Y, X, Y_aux, beta, gamma, neig):
    N = Y_aux.shape[0]
    hess_all = hessianLogH_vmap(Y_aux, X, beta, gamma, neig)
    grad_all = gradLogH_vmap(Y_aux, X, beta, gamma, neig)
    T1 = jnp.mean(hess_all, axis=0)
    outer_vmapped = jax.vmap(lambda g: jnp.outer(g, g))
    T2 = jnp.mean(outer_vmapped(grad_all), axis=0)
    grad_mean = jnp.mean(grad_all, axis=0)
    T3 = jnp.outer(grad_mean, grad_mean)
    return T1 + T2 - T3
   
  
def RappHessc(Y, X, Y_aux, beta, gamma, neig):
    result = appHessc(Y, X, Y_aux, beta, gamma, neig)
    return numpy.array(result)

##8. H element of d


@jax.jit
def Hd(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    T1 = hessianLogH(Y, X, beta, gamma, neig)
    T2 = appHessc(Y, X, Y_aux, beta, gamma, neig)
    T3 = hessianLogPrior(beta, gamma, sdBeta)
    A = T1 - T2 + T3
    tril_indices = jnp.tril_indices(A.shape[0])
    result = A[tril_indices]
    return result
  
def RHd(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    result= Hd(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    return numpy.array(result)

###9. Calculate d vector for approx curvature diagnostic

def dFun(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    H = Hd(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    J = Jd(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    tot=H+J
    return numpy.array(tot)
  
def RdFun(Y, X, Y_aux, beta, gamma, neig, sdBeta):
    result= dFun(Y, X, Y_aux, beta, gamma, neig, sdBeta)
    return numpy.array(result)
