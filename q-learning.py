"""The implementation of Q-learning and Double Q-learning algorithms for
   a simplified grid world problem.
"""
import numpy as np
import matplotlib.pyplot as plt
import random


# Markov Decision Process
class MDP(object):

    def __init__(self, mu=-0.1, sigma=1, k=5):
        self.mu = mu
        self.sigma = sigma
        self.s = None
        self.k = k

    def reset(self):
        self.s = 0
        return self.s

    def step(self, a, left_count):
        assert(self.s != 2)
        # s==0 means circle A, s==1 means circle B, s==2 means terminal
        # a==0 means left, a==1 means right
        if self.s == 0 and a == 1:
            self.s = 2
            r = 0
            t = True
        elif self.s == 0 and a == 0:
            self.s = 1
            r = 0
            t = False
            left_count += 1
        else:
            self.s = 2
            r = np.random.normal(self.mu, self.sigma)
            t = True

        return (r, self.s, t, left_count)


k = 2
env = MDP(k=2)
a_big_number = 100
epsilon = 0.1
alpha = 0.1
gamma = 0.99
num = []
num_double = []
Q = np.zeros([3, k + 2])
Q[0, 2:k + 2] = -a_big_number
Q[1, 0:2] = -a_big_number


def Q_learning(Q, n):
    count = 0
    left_count = 0
    for i in range(n):
        s = env.reset()
        t = False
        while(not t):
            # TODO: Q-learning Algorithm
            n = random.uniform(0, 1)
            if epsilon < n:
                a = np.argmax(Q[s])
            elif s == 0:
                a = np.random.randint(0, 2)
            elif s == 1:
                a = np.random.randint(2, 4)
            r, s_prime, t, left_count = env.step(a, left_count)
            count += 1
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])
            s = s_prime
        num.append(left_count / count)
    return Q


Q1, Q2 = np.zeros([3, k + 2]), np.zeros([3, k + 2])
Q1[0, 2:k + 2], Q2[0, 2:k + 2] = -a_big_number, -a_big_number
Q1[1, 0:2], Q2[1, 0:2] = -a_big_number, -a_big_number


def double_Q_learning(Q1, Q2, n):
    count = 0
    left_count = 0
    for i in range(n):
        s = env.reset()
        t = False
        while(not t):
            # TODO: Q-learning Algorithm
            n = random.uniform(0, 1)
            if epsilon < n:
                a = np.argmax(Q1[s] + Q2[s])
            elif s == 0:
                a = np.random.randint(0, 2)
            elif s == 1:
                a = np.random.randint(2, 4)
            r, s_prime, t, left_count = env.step(a, left_count)
            count += 1
            if random.uniform(0, 1) > 0.5:
                Q1[s, a] += alpha * (r + gamma * Q2[s_prime, np.argmax(Q1[s_prime])] - Q1[s, a])
            else:
                Q2[s, a] += alpha * (r + gamma * Q1[s_prime, np.argmax(Q2[s_prime])] - Q2[s, a])
            s = s_prime
        num_double.append(left_count / count)
    return Q1 + Q2


# print(num)
print(Q_learning(Q, 400))
print(double_Q_learning(Q1, Q2, 400))
plt.plot(range(len(num)), num)
plt.plot(range(len(num_double)), num_double)
plt.show()
