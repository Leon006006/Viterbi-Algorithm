import numpy as np


class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def viterbi(self, Observable):
        array = np.zeros((self.A.shape[0], Observable.shape[1], 2))
        numb_status = array.shape[0]
        T = array.shape[1]

        for x in range(0, numb_status):
            array[x, 0, 0] = self.Pi[x] * self.B[x, Observable[0, 0]]
            array[x, 0, 1] = 20

        for t in range(1, T):
            for status in range(0, numb_status):
                v = np.zeros((numb_status, 1))
                for pre_status in range(0, numb_status):
                    v[pre_status, 0] = array[pre_status, t-1, 0] * self.A[pre_status, status] * \
                                       self.B[status, Observable[0, t]]
                array[status, t, 0] = max(v)
                array[status, t, 1] = np.argmax(v)

        highest_prop = np.argmax(array[:, T-1, 0])
        probability_observation = array[highest_prop, T-1, 0]

        print("Highest probability for observation-sequence: {}%".format(np.round(probability_observation * 100, 2)))

        status_seq = np.zeros((1, T))
        status_seq[0, T-1] = highest_prop

        for t in range(1, T):
            g = status_seq[0, T-t].astype(int)
            status_seq[0, T-t-1] = array[g, T-t, 1]

        print("Status sequenz for highest probability: {}".format(status_seq))