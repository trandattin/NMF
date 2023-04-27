import numpy as np
import time

def lee_seung_algorithm(V,W,H,times=100,epsilon=1e-05):
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
    # Update W
    numerator_w = np.dot(V, H)
    denominator_w = np.dot(np.dot(W, np.transpose(H)), H) + epsilon
    W *= numerator_w / denominator_w

    # Update H
    numerator_h = np.dot(np.transpose(V), W)
    denominator_h = np.dot(np.dot(H, np.transpose(W)), W) + epsilon
    H *= numerator_h / denominator_h
	
    # Calculate the error and append it to the list
    error = np.linalg.norm(V - np.dot(W,H.T))
    errors.append(error)
    times_list.append(time.time() - start_time)
  V = W @ H.T
  return V, W, H.T, [times_list,errors]