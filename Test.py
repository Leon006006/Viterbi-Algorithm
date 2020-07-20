import numpy as np
import HMM as HMM

# Initialize required matrices and the HMM
A = np.array([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
B = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.4, 0.4]])
Pi = np.array([[0.4], [0.3], [0.3]])
HMM1 = HMM.HMM(A, B, Pi)

# Initialize observed vector
Observation = np.array([[0, 1, 0, 2, 2, 2]])

# Run Viterbi algorithm
status_seq = HMM1.viterbi(Observation)

print("Status sequence with highest probability: {}".format(status_seq))
