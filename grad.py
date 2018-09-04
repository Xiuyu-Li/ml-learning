"""Use gradient descent to fit a linear function. The basic euqation
   representation of gradient descent is used in this program.
"""
import numpy as np

a = 0
b = 0
k = 0.00001

x, y = np.loadtxt('/Users/pro/Desktop/data.txt')

sumx = np.sum(x)
sumy = np.sum(y)
sumxy = np.sum(x * y)
sumx2 = np.sum(x**2)

grad_a = 1 / 100 * (a * sumx2 + b * sumx - sumxy)
grad_b = 1 / 100 * (a * sumx + 100 * b - sumy)
print(grad_a, grad_b)

while abs(grad_a) >= 0.0001 or abs(grad_b) >= 0.0001:
    a -= k * grad_a
    b -= k * grad_b
    grad_a = 1 / 100 * (a * sumx2 + b * sumx - sumxy)
    grad_b = 1 / 100 * (a * sumx + 100 * b - sumy)

print(a, b)
print(grad_a, grad_b)
