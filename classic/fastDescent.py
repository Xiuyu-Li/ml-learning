"""Use gradient descent to fit a linear function. The matrix representation of
   gradient descent is used in this program.
"""
import numpy as np

x, y = np.loadtxt('/Users/pro/Desktop/data.txt')

n = 100
A = np.array([np.ones(n), x])
A = A.transpose()
b = y

x = np.array([0, 0])

epsilon = 1e-6
delta = 1e-4
T = 0
while(True):
    G = A.transpose() @ (A @ x - b)
    nG = np.linalg.norm(G)
    # print('Gradient norm: %f' % nG)
    if nG < delta:
        break
    x = x - epsilon * G
    T = T + 1
print((x, T))
