"""The Action-value method and UCB (Upper-Confidence-Bound) method for
   finding the best strategie to play Multi-armed Bandit in non-stable
   condition.
"""
import numpy as np
import math


def generator(a):
    mu = np.array([3, 4, 0, 1, 2, 3.5, 2.5, 3, 1.5, 0.5])
    d = (np.random.normal(0., 0.1, 10) + mu)
    return d[a]


def rewardEstimator(a, num):
    if num == 1:
        return generator(a)
    else:
        alpha = 0.5
        secondpart = 0
        for i in range(0, num - 1):
            secondpart += alpha * (1 - alpha)**(num - 1 - i) * generator(a)
        return (1 - alpha) ** (num - 1) * rewardEstimator(a, 1) + secondpart


# the Action-value method for selecting the next step
def actionValue(table):
    if table == [0]:
        return np.random.randint(0, 10)
    else:
        a = table.index(max(table))
        i = np.random.randint(0, 10)
        if i <= 4:
            return a
        else:
            return np.random.randint(0, 10)


# the UCB (Upper-Confidence-Bound) method for selecting the next step
countA = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}


def UCB(table, num):
    if table == [0]:
        return np.random.randint(0, 10)
    else:
        listA = []
        for i in range(10):
            second = 0
            if countA[i] == 0:
                second = 100000
            else:
                second = 0.5 * (math.log(num) / countA[i])**0.5
            listA.append(table[i] + second)
    return listA.index(max(listA))


Q_table = [0] * 10
rounds = 1
while(rounds <= 5000):
    # a = actionValue(Q_table)
    a = UCB(Q_table, rounds - 1)
    countA[a] += 1
    Q_table[a] = rewardEstimator(a, rounds)
    rounds += 1
print(Q_table)
