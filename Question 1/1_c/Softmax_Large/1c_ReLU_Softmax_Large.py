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
        a = a.reshape((784, 1))
        for i in range(len(self.weights)-1):
            w = self.weights[i]
            b = self.biases[i]
            a = relu(np.dot(w, a) + b)
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
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
        x = np.array(x).reshape((len(x),1))
        y = np.array(y).reshape((len(y), 1))

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                relu_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        X = []
        Y = []
        for tx, ty in test_data:
            X.append(tx)
            Y.append(ty)
        ans = []
        for tx, ty in test_data:
            a = self.feedforward(tx)
            ans.append(a)
        Y_comp = [[0]*10 for _ in range(len(ans))]
        for ctr in range(len(ans)):
            Y_comp[ctr][np.argmax(ans[ctr])] = 1
        result = np.sum(int(x==x_comp.tolist()) for x,x_comp in zip(Y_comp,Y))
        return result

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    x[x < 0] = 1
    return np.log(x)

def relu_prime(z):
    res = []
    for item in z:
        if item>0:
            res.append([1.0])
        else:
            res.append([0.0])
    return np.array(res)


def read(dataset, path):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


def data_load():
    training = list(read(dataset='training', path = '/home/iiitd/Music/HW-3-NN/dataset'))
    testing = list(read(dataset='testing', path = '/home/iiitd/Music/HW-3-NN/dataset'))

    #Training dataset
    train_Y, train_X = zip(*training)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    train_X = train_X.reshape(train_X.shape[0], -1)
    X_new = np.zeros(train_X.shape)
    for i in range(len(train_X)):
        for j in range(len(train_X[i])):
            X_new[i][j] = train_X[i][j]/255.0
    train_X = X_new


    Y_new = np.zeros((len(train_Y), 10))
    for i in range(len(train_Y)):
        Y_new[i][train_Y[i]] = 1
    train_Y = Y_new

    #Testing dataset
    test_Y, test_X = zip(*testing)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    test_X = test_X.reshape(test_X.shape[0], -1)
    X_new = np.zeros(test_X.shape)
    for i in range(len(test_X)):
        for j in range(len(test_X[i])):
            X_new[i][j] = test_X[i][j]/255.0
    test_X = X_new

    Y_new = np.zeros((len(test_Y), 10))
    for i in range(len(test_Y)):
        Y_new[i][test_Y[i]] = 1
    test_Y = Y_new

    training = zip(train_X, train_Y)
    testing = zip(test_X, test_Y)
    return training, testing

def plot_graph(epoch, accu, count):
    pyplot.xlabel('epoch')
    pyplot.ylabel('Accuracy')
    pyplot.title('Question 1c_relu_softmax_large'+str(count))
    pyplot.plot(epoch, accu)
    pyplot.savefig('1c_relu_softmax_large'+str(count)+'.jpg', dpi=100)


train_data, test_data = data_load()


learing_rates = [0.001, 0.003, 0.01, 0.1]
f = open('1c_relu_softmax_largeAccuracy.txt', 'w')
a = [784, 100, 50, 10]
objects = []
pos = -1
max_learn = 0.0
for i in range(len(learing_rates)):
    NNobj = Network(a)
    objects.append(NNobj)
    epoch, accu = NNobj.SGD(train_data, 15, 10, learing_rates[i], test_data)
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
with open('1c_relu_softmax_large.pkl', 'wb') as output:
    pickle.dump(objects[pos], output, pickle.HIGHEST_PROTOCOL)