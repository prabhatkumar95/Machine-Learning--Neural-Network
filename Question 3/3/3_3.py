import os
import h5py
import struct
import pickle
import numpy as np
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier


def read(dataset = "training", path = "."):
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

    get_img = lambda idx: (img[idx], lbl[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


training = list(read(dataset='training', path = '/home/arpn/Machine Learning/HW-3-NN/dataset'))
testing = list(read(dataset='testing', path = '/home/arpn/Machine Learning/HW-3-NN/dataset'))

#Training dataset
train_X, train_Y = zip(*training)
train_X = np.array(train_X)
train_Y = np.array(train_Y)
train_X = train_X.reshape((train_X.shape[0], -1))

#Testing dataset
test_X, test_Y = zip(*testing)
test_X = np.array(test_X)
test_Y = np.array(test_Y)
test_X = test_X.reshape((test_X.shape[0], -1))


f = open('3_1.txt', 'w')

alphaList = [0.001, 0.01, 0.1, 0.2, 0.8]
scores = []
for alp in alphaList:
    nn_obj = MLPClassifier(random_state=1, hidden_layer_sizes=(600, 500, 400, 300, 200, 100, 50), activation='relu',
                           early_stopping=True, verbose=True, alpha=alp, solver='adam')
    nn_obj.fit(train_X, train_Y)
    predict_y = nn_obj.predict(test_X)
    tempscore = nn_obj.score(test_X, test_Y)
    scores.append(tempscore)

optscore = 0
ind = -1

for i in range(len(scores)):
    f.write('For the alpha value '+str(alphaList[i])+' the accuracy is '+str(scores[i])+'\n')
    if optscore < scores[i]:
        optscore = scores[i]
        ind = i

f.write('\n\n\nMaximum score is '+str(optscore))


nn_obj = MLPClassifier(random_state=1, hidden_layer_sizes = (600, 500, 400, 300, 200, 100, 50), activation='relu', early_stopping=True, verbose=True, alpha=alp, solver='adam')
nn_obj.fit(train_X, train_Y)
pyplot.xlabel('epoch')
pyplot.ylabel('Accuracy')
pyplot.title('Question 3_1')
pyplot.plot(nn_obj.validation_scores_)
pyplot.savefig('3_1.jpg', dpi = 100)
f.close()

with open('model_3_1.pkl', 'wb') as output:
    pickle.dump(nn_obj, output, pickle.HIGHEST_PROTOCOL)
