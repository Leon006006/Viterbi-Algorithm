import numpy as np


class HMM:
    def __init__(self, A, B, Pi):
        """
        :param A: Transition probabilities
        :param B: Emission probabilities
        :param Pi: Starting probabilities
        """
        self.A = A
        self.B = B
        self.Pi = Pi

    def viterbi(self, Observable):
        """
        :param Observable: Observed sequence
        :return: Most probable status sequence
        """

        # Table for probabilities and predecessor
        array = np.zeros((self.A.shape[0], Observable.shape[1], 2))
        # Number of statuses
        numb_status = array.shape[0]
        # Number of points in time
        T = array.shape[1]

        # Compute probabilities for first point in time
        for x in range(0, numb_status):
            array[x, 0, 0] = self.Pi[x] * self.B[x, Observable[0, 0]]
            array[x, 0, 1] = -1     # Predecessor on "random" value

        # Compute all necessary probabilities
        for t in range(1, T):
            for status in range(0, numb_status):
                v = np.zeros((numb_status, 1))
                for pre_status in range(0, numb_status):
                    v[pre_status, 0] = array[pre_status, t-1, 0] * self.A[pre_status, status] * \
                                       self.B[status, Observable[0, t]]
                # Find maximum probability for given observation
                array[status, t, 0] = max(v)
                array[status, t, 1] = np.argmax(v)

        # Get highest probability of observation
        highest_prop = np.argmax(array[:, T-1, 0])
        probability_observation = array[highest_prop, T-1, 0]
        print("Highest probability for observation-sequence: {}%".format(np.round(probability_observation * 100, 2)))

        # Initialize sequence of states vector
        status_seq = np.zeros((1, T))
        status_seq[0, T-1] = highest_prop

        # Compute and save status sequence
        for t in range(1, T):
            g = status_seq[0, T-t].astype(int)
            status_seq[0, T-t-1] = array[g, T-t, 1]

        return status_seq
