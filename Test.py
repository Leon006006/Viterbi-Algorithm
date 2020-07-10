import numpy as np
import HMM as HMM


A = np.array([[0.8, 0.2], [0.1, 0.9]])
B = np.array([[0.7, 0.3], [0.2, 0.8]])
Pi = np.array([[0.5], [0.5]])
Observation = np.array([[0, 1, 0]])

HMM1 = HMM.HMM(A, B, Pi)

HMM1.viterbi(Observation)