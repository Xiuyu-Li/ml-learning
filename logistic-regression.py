import numpy as np
import math
import random
import time
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


def sigmoid(x):
    """the sigmoid function used in logistics regression"""
    return 1 / (1 + np.exp(-x))


def mini_batch_grad_descent(X, y, mini_batch_size=64, seed=0):
    """an implementation of the mini-batch gradient descent to improve the convergence speed"""
    np.random.seed(seed)
    m = X.shape[0]
    num_mini_batches = m // mini_batch_size
    mini_batches = []
    permutation = list(np.random.permutation(m))

    # shuffle the order of inputs and outputs
    X_shuffled = X[permutation, :]
    y_shuffled = y[permutation, :]
    for i in range(num_mini_batches):
        X_mini_batch = X_shuffled[i * mini_batch_size: (i + 1) * mini_batch_size, :]
        y_mini_batch = y_shuffled[i * mini_batch_size: (i + 1) * mini_batch_size, :]
        mini_batches.append((X_mini_batch, y_mini_batch))

    # handle the condition when the size of the last mini-batch is less than the mini_batch_size
    if m & mini_batch_size != 0:
        X_mini_batch = X_shuffled[num_mini_batches * mini_batch_size: m, :]
        y_mini_batch = y_shuffled[num_mini_batches * mini_batch_size: m, :]
        mini_batches.append((X_mini_batch, y_mini_batch))
    return mini_batches


if __name__ == '__main__':
    # use the sklearn breast cancer dataset
    cancer = datasets.load_breast_cancer()
    X = cancer.data[:, :]

    # add a constant term to the left
    X = np.insert(X, 0, 1, axis=1)
    y = cancer.target

    # check the shape of X and y
    # print(X.shape)
    # print(y.shape)

    # get an overview of the content of X
    # print(X)

    # split the training set and the testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # print(X_train.shape)
    # print(y_train.shape)

    # change the shape of y from (n ,) to (n, 1)
    y_train = y_train.reshape(np.shape(X_train)[0], 1)

    # generate the coefficient
    theta = np.ones((1, np.shape(X_train)[1]))
    # print(theta.shape)

    """use gradient descent to get the optimal theta"""
    alpha = 1e-3 # step length
    epsilon = 1e-3 # the threshold value of gradient to break the iteration
    T = 0
    theta_temp = theta
    start_time = time.time()
    while(True):
        G = 1 / 100 * X_train.transpose() @ (sigmoid(X_train @ theta_temp.transpose()) - y_train)
        nG = np.linalg.norm(G)
        # also break the iteration if it has run more than 300000 times
        if nG < epsilon or T > 300000:
            break
        theta_temp = theta_temp - epsilon * G.transpose()
        T = T + 1
    # print((sigmoid(X_train @ theta_temp.transpose()))
    print('{} {} {} {}{}'.format("Iteration times:", T, "Computation time:", time.time() - start_time, 's'))

    """test the model"""
    count = 0 # the number of correct predictions

    # the output of the model using testing set
    for i in range(np.shape(X_test)[0]):
        temp = 0
        if sigmoid(X_test[i] @ theta_temp.transpose()) >= 0.5:
            temp = 1
        if temp == y_test[i]:
            count = count + 1
    print('Logistic Regression Accuracy (Gradient Descent):',  count / np.shape(y_test)[0])

    """plot the result (it is kind of useless)"""
    # plt.figure()
    # plt.subplot(211)
    # plt.scatter(X_test[:, 1], y_points, color='red')
    # plt.subplot(212)
    # plt.scatter(X_test[:, 1], y_test, color='green')
    # plt.show()

    """use mini-batch gradient descent"""
    seed = 0
    alpha = 1e-3
    epsilon = 1e-3
    T = 0
    start_time = time.time()
    theta_temp = theta

    #  break the iteration if it has run more than 300000 times
    while(T < 300000):
        # use random seed to do the shuffle every time
        seed = seed + random.randint(1, 100)
        mini_batches = mini_batch_grad_descent(X_train, y_train, 64, seed)
        for mini_batch in mini_batches:
            G = 1 / 64 * mini_batch[0].transpose() @ (sigmoid(mini_batch[0] @ theta_temp.transpose()) - mini_batch[1])
            nG = np.linalg.norm(G)
            # do not end the iteration if it has only run less than 20000 times
            if nG < epsilon and T > 20000:
                break
            theta_temp = theta_temp - epsilon * G.transpose()
            T = T + 1
        if nG < epsilon and T > 20000:
            break
    print('{} {} {} {}{}'.format("Iteration times:", T, "Computation time:", time.time() - start_time, 's'))

    """test the accuracy"""
    count = 0
    for i in range(np.shape(X_test)[0]):
        temp = 0
        if sigmoid(X_test[i] @ theta_temp.transpose()) >= 0.5:
            temp = 1
        if temp == y_test[i]:
            count = count + 1
    print("Logistic Regression Accuracy (Mini-batch Gradient Descent):", count / np.shape(y_test)[0])
