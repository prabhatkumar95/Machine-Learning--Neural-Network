import os
import h5py
import pickle
import struct
import numpy as np
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, '/home/arpn/Machine Learning/HW-3-NN/dataset/train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, '/home/arpn/Machine Learning/HW-3-NN/dataset/train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, '/home/arpn/Machine Learning/HW-3-NN/dataset/t10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, '/home/arpn/Machine Learning/HW-3-NN/dataset/t10k-labels.idx1-ubyte')
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


data = list(read())
x = []
y = []

for i in range(len(data)):
    x.append(np.array(data[i][1]).flatten())
    y.append(data[i][0])

# print y
x = np.array(x)
y = np.array(y)

data_x = np.array([i.flatten() for i in x])

train_x, test_x, train_y, test_y = train_test_split(data_x, y, test_size=0.2)

f = open('2b.txt', 'w')

alphaList = [0.001, 0.01, 0.1, 0.2, 0.8, 1.0]
scores = []
for alp in alphaList:
    nn = MLPClassifier((100, 50), activation='logistic', early_stopping=True, verbose=True, alpha=alp)
    nn.fit(train_x, train_y)
    predict_y = nn.predict(test_x)
    tempscore = nn.score(test_x, test_y)
    scores.append(tempscore)

optscore = 0
ind = -1

for i in range(len(scores)):
    f.write('For the alpha value '+str(alphaList[i])+' the accuracy is '+str(scores[i])+'\n')
    if optscore < scores[i]:
        optscore = scores[i]
        ind = i

f.write('\n\n\nMaximum score is '+str(optscore))


nn = MLPClassifier((100, 50), activation='logistic', early_stopping=True, verbose=True, alpha=alphaList[ind])
nn.fit(train_x, train_y)
pyplot.xlabel('epoch')
pyplot.ylabel('Accuracy')
pyplot.title('Question 2b')
pyplot.plot(nn.validation_scores_)
pyplot.savefig('2b.jpg', dpi = 100)
f.close()


with open('model_2b.pkl', 'wb') as output:
    pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)
