import tensorflow as tf
import numpy as np
import random

# import mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    dx = sigmoid(x)
    return dx * (1.0 - dx)


def get_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """an implementation to get the mini-batches of the batch gradient descent to improve the convergence speed"""
    np.random.seed(seed)
    m = X.shape[0]
    num_mini_batches = m // mini_batch_size
    mini_batches = []
    permutation = list(np.random.permutation(m))

    # shuffle the order of inputs and outputs
    X_shuffled = X[permutation, :]
    Y_shuffled = Y[permutation]
    for i in range(num_mini_batches):
        X_mini_batch = X_shuffled[i * mini_batch_size: (i + 1) * mini_batch_size, :]
        y_mini_batch = Y_shuffled[i * mini_batch_size: (i + 1) * mini_batch_size]
        x_list = [np.reshape(x, (784, 1)) for x in X_mini_batch]
        y_list = []

        for y_value in y_mini_batch:
            y = np.zeros((10, 1))
            y[y_value] = 1.0
            y_list.append(y)
        data = zip(x_list, y_list)
        data = list(data)
        mini_batches.append(data)

    # handle the condition when the size of the last mini-batch is less than the mini_batch_size
    if m % mini_batch_size != 0:
        X_mini_batch = X_shuffled[num_mini_batches * mini_batch_size: m, :]
        y_mini_batch = Y_shuffled[num_mini_batches * mini_batch_size: m]
        x_list = [np.reshape(x, (784, 1)) for x in X_mini_batch]
        y_list = []
        for y_value in y_mini_batch:
            y = np.zeros((10, 1))
            y[y_value] = 1.0
            y_list.append(y)
        data = zip(x_list, y_list)
        data = list(data)
        mini_batches.append(data)
    return mini_batches


class Network(object):
    """the fully connected neural network"""
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        """feedforward to compute loss"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """backpropagate to compute gradients"""
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        activations = []
        # the input will act as the activation of the weight in the first layer
        activations.append(x)
        zs = []
        l = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[l]) + b
            zs.append(z)
            activations.append(sigmoid(z))
            l = l + 1
        delta = (activations[-1] - y) * (sigmoid_prime(zs[-1]))
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return zip(grad_b, grad_w)

    def update(self, mini_batch, alpha):
        """update weights and biases with gradients derived by backpropagation"""
        grad_b_sum = [np.zeros(b.shape) for b in self.biases]
        grad_w_sum = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            l = 0
            for grad_b, grad_w in self.backprop(x, y):
                grad_b_sum[l] = grad_b_sum[l] + grad_b
                grad_w_sum[l] = grad_w_sum[l] + grad_w
                l = l + 1
        self.biases = [b - alpha / len(mini_batch) * grad_b for b, grad_b in zip(self.biases, grad_b_sum)]
        self.weights = [w - alpha / len(mini_batch) * grad_w for w, grad_w in zip(self.weights, grad_w_sum)]

    def batch_grad_descent(self, x_train, y_train, alpha, epoches, test_data):
        """batch gradient descent algorithm"""
        seed = 0
        for i in range(epoches):
            seed = seed + random.randint(1, 100)
            mini_batches = get_mini_batches(x_train, y_train, 64, seed)
            for mini_batch in mini_batches:
                self.update(mini_batch, alpha)
            count = 0
            for (x, y) in test_data:
                if np.argmax(self.feedforward(x)) == y:
                    count = count + 1
            print('{} {}{} {}{}'.format("Epoch", i, ":", format(count / len(test_data) * 100.0, '.2f'), "%"))


if __name__ == '__main__':
    test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
    test_data = list(zip(test_inputs, y_test))
    net = Network([784, 30, 10])
    net.batch_grad_descent(x_train, y_train, 3.0, 10, test_data)
