import os
import h5py
import struct
import pickle
import random
import numpy as np
from random import shuffle
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        epochlist = []
        acculist = []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            acculist.append(float(self.evaluate(test_data))/float(n_test))
            epochlist.append(j)
        return epochlist,acculist

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def data_load():
    file = h5py.File('/home/arpn/Machine Learning/HW-3-NN/dataset_partA.h5', 'r')

    X = file['X'][:]
    X = X.reshape((X.shape[0],-1))
    Y = file['Y'][:]
    for value in range(0, len(Y)):
        if Y[value] == 9:
            Y[value] = 1.0
        else:
            Y[value] = 0.0
    X_temp = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X_temp[i][j] = X[i][j]/255.0
    X = X_temp

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)
    a = []

    a.append(X_train)
    a.append(Y_train)
    x = zip(*zip(a))
    train_data = x[0]

    b = []
    b.append(X_test)
    b.append(Y_test)
    x = zip(*zip(b))
    test_data = x[0]

    return train_data, test_data


def data_conversion():
    train_data, test_data = data_load()
    training_inputs = [np.reshape(xx, (784, 1)) for xx in train_data[0]]
    training_results = [vectorized_result(y) for y in train_data[1]]
    train_data = zip(training_inputs, training_results)

    test_inputs = [np.reshape(xx, (784, 1)) for xx in test_data[0]]

    test_data = zip(test_inputs, test_data[1])

    return train_data, test_data


def vectorized_result(i):
    a = np.zeros(shape=(2, 1))
    a[i] = 1
    return a

def plot_graph(epoch, accu, count):
    pyplot.xlabel('epoch')
    pyplot.ylabel('Accuracy')
    pyplot.title('Question 1.a'+str(count))
    pyplot.plot(epoch, accu)
    pyplot.savefig('1a_'+str(count)+'.jpg', dpi=100)


train_data, test_data = data_conversion()


learing_rates = [0.001, 0.01, 0.1, 0.8, 0.9]
f = open('1a_Accuracy.txt', 'w')
a = [784, 100, 50, 2]
objects = []
pos = -1
max_learn = 0.0
for i in range(len(learing_rates)):
    NNobj = Network(a)
    objects.append(NNobj)
    epoch, accu = NNobj.SGD(train_data, 30, 10, learing_rates[i], test_data)
    plot_graph(epoch, accu, i+1)
    f.write('For learning rate: ' + str(learing_rates[i]) + '\n')
    f.write('____________________________________________________________\n')
    f.write('Epoch No \t Accuracy\n')
    for cnt in range(len(accu)):
        f.write(str(cnt + 1) + '\t' + str(accu[cnt]) + '\n')
    f.write('Maximum for this learning rate: ' + str(max(accu)) + '\n\n\n')
    if max_learn < max(accu):
        max_learn = max(accu)
        pos = i

f.close()
with open('1a.pkl', 'wb') as output:
    pickle.dump(objects[pos], output, pickle.HIGHEST_PROTOCOL)