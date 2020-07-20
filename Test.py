import numpy as np
import HMM as HMM

# Initialize required matrices and the HMM
A = np.array([[0.8, 0.2], [0.1, 0.9]])
B = np.array([[0.7, 0.3], [0.2, 0.8]])
Pi = np.array([[0.5], [0.5]])
HMM1 = HMM.HMM(A, B, Pi)

# Initialize observed vector
Observation = np.array([[0, 1, 0, 1, 0, 1]])

# Run Viterbi algorithm
status_seq = HMM1.viterbi(Observation)

print("Status sequence with highest probability: {}".format(status_seq))
