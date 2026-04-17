"""
Hidden Marcov Model framework. Uses forward algorithm for predictions and Baum Welch for training.
Used for dynamic classifier classifications.
"""
import numpy as np

class HMM:
    # Initializes the HMM
    def __init__(self, n_states, n_obs_states=10):
        self.n = n_states
        self.m = n_obs_states
        self.pi = np.random.dirichlet(np.ones(self.n))
        # transition probabilities
        self.A = np.array([np.random.dirichlet(np.ones(self.n)) for _ in range(self.n)])
        # emission probabilities
        self.B = np.array([np.random.dirichlet(np.ones(self.m)) for _ in range(self.n)])

    # Returns the probability of a given sequence
    def forward(self, obs_seq):
        T = len(obs_seq)
        if T <= 1:
            # exit
            return None, None, None
        alpha = np.zeros((T, self.n))
        scale = np.zeros(T)
        # initialization
        for i in range(self.n):
            alpha[0, i] = self.pi[i] * self.B[i, obs_seq[0]]
        scale[0] = sum(alpha[0])
        if scale[0] == 0:
            scale[0] = 1e-300
        alpha[0] /= scale[0]

        # reccurence
        for t in range(1, T):
            for i in range(self.n):
                alpha[t, i] = self.B[i, obs_seq[t]] * np.dot(alpha[t-1], self.A[:, i])
            scale[t] = sum(alpha[t])
            if scale[t] == 0:
                scale[t] = 1e-300
            alpha[t] /= scale[t]

        return np.sum(np.log(scale + 1e-300)), alpha, scale
    
    def backward(self, obs_seq, scale):
        T = len(obs_seq)
        if T <= 1:
            # exit
            return None, None, None
        beta = np.zeros((T, self.n))
        beta[T - 1, :] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(self.n):
                beta[t, i] = np.dot(self.A[i] * self.B[:, obs_seq[t + 1]], beta[t+1])
            beta[t] /= scale[t + 1]
        return beta
    
    # Trains the HMM
    def baum_welch(self, data, n_iter=50, print_output=False):
        log_probs = []
        for _ in range(n_iter):
            pi_acc  = np.zeros(self.n)
            A_num   = np.zeros((self.n, self.n))
            A_den   = np.zeros(self.n)
            B_num   = np.zeros((self.n, self.m))
            B_den   = np.zeros(self.n)

            for seq in data:
                log_prob, alpha, scale = self.forward(seq)
                if log_prob is None:
                    continue
                beta = self.backward(seq, scale)
                T = len(seq)

                gamma = alpha * beta
                row_sums = gamma.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1e-300
                gamma /= row_sums

                # probabilities of being in state i at time t and state j at time t + 1
                xi = np.zeros((T - 1, self.n, self.n))
                for t in range(T - 1):
                    for i in range(self.n):
                        for j in range(self.n):
                            xi[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, seq[t + 1]] * beta[t + 1, j]
                    xi_sum = xi[t].sum()
                    if xi_sum > 0:
                        xi[t] /= xi_sum

                pi_acc += gamma[0]
                A_num  += xi.sum(axis=0)
                A_den  += gamma[:-1].sum(axis=0)
                B_den  += gamma.sum(axis=0)
                for k in range(self.m):
                    mask = np.array(seq) == k
                    B_num[:, k] += gamma[mask].sum(axis=0)

            # update probabilities
            self.pi = pi_acc / (pi_acc.sum() + 1e-300)          # + 1e-300 prevents division by 0
            self.A  = A_num / (A_den[:, np.newaxis] + 1e-300)
            self.B  = B_num / (B_den[:, np.newaxis] + 1e-300)

            avg_log_prob = np.mean([
                self.forward(seq)[0] for seq in data
                if self.forward(seq)[0] is not None
            ])
            log_probs.append(avg_log_prob)

            if (print_output):
                print(f"Iter {_+1}: avg log prob = {avg_log_prob:.3f}")
        return log_probs


'''
h = HMM(5, 10)
sequences = [
    [8, 7, 6, 4, 2, 1],
    [9, 8, 7, 5, 3, 1],
    [8, 6, 5, 3, 2, 0],
    [9, 7, 6, 4, 2, 1],
]
prob_before, _, _ = h.forward(sequences[0])
h.baum_welch(sequences, n_iter=50)

prob_high, _, _ = h.forward(sequences[0])
prob_mid, _, _ = h.forward([7, 5, 4, 2, 1, 1])
prob_low, _, _ = h.forward(list(reversed(sequences[0])))
print(f"high log prob: {prob_high: .3f}")
print(f"mid log prob: {prob_mid: .3f}")
print(f"low log prob: {prob_low: .3f}")
'''
