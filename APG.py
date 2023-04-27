import numpy as np
import time

def alternating_projected_gradient(V,W,H,times=100):
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
    # Compute gradient when H fixed
    dev_W =  np.dot(W, np.dot(np.transpose(H), H)) - np.dot(V,H)
    numerator_epsilon_W = np.linalg.norm(dev_W , "fro")**2
    denominator_epsilon_W = np.linalg.norm(np.dot(dev_W,np.transpose(H)), "fro")**2
    epsilon_W = numerator_epsilon_W / denominator_epsilon_W

    # Update H
    W -= epsilon_W*dev_W
    # Set all negative value in H to 0
    W = np.maximum(0,W)

    # Compute gradient when W fixed
    dev_H =  np.dot(H, np.dot(np.transpose(W), W)) - np.dot(np.transpose(V),W)
    numerator_epsilon_H = np.linalg.norm(dev_H, "fro")**2
    denominator_epsilon_H = np.linalg.norm(np.dot(dev_H,np.transpose(W)), "fro")**2
    epsilon_H = numerator_epsilon_H / denominator_epsilon_H

    # Update H
    H -= epsilon_H*dev_H
    # Set all negative value in H to 0
    H = np.maximum(0,H)

    # Calculate the error and append it to the list
    error = np.linalg.norm(V - np.dot(W,H.T))
    errors.append(error)
    times_list.append(time.time() - start_time)
  V = W @ H.T
  return V, W, H.T, [times_list,errors]