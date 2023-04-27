import numpy as np
import time

def HALS_RRI(V,W,H,r,times=100):  
  W = np.copy(W)
  H = np.copy(H)
  V = np.copy(V)
  start_time = time.time()
  times_list  = []
  errors = []
  while True:
    # Check time limit
    if time.time() - start_time >= times:
        break
    # Update each basis vector in turn
    for k in range(r):        
      # Calculate the residue matrix
      R = V - np.dot(W,np.transpose(H)) + np.outer(W[:,k],H[:,k])
      # Update the k-th basis vector
      K = np.dot(np.transpose(R), W[:, k])
      numerator = np.maximum(0,K)
      denominator = np.linalg.norm(W[:,k])**2
      if np.all(numerator == 0):
        H[:,k] = np.full(H.shape[0], 0)
      else:
        H[:,k] = numerator / denominator
            
      # Update the corresponding coefficients
      K = np.dot(R,H[:,k])
      numerator = np.maximum(0,K)
      denominator = np.linalg.norm(H[:,k])**2
      if np.all(numerator == 0):
        W[:,k] = np.full(W.shape[0], 0)
      else:
        W[:,k] = numerator / denominator
    # Calculate the error and append it to the list
    error = np.linalg.norm(V - np.dot(W,H.T))
    errors.append(error)
    times_list.append(time.time() - start_time)
  V = W @ H.T
  return V, W, H.T, [times_list, errors]