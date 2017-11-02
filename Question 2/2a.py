import h5py
import pickle
import struct, os
import numpy as np
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y

x, y = load_h5py('/home/arpn/Machine Learning/HW-3-NN/dataset_partA.h5')

data_x = np.array([i.flatten() for i in x])

train_x, test_x, train_y, test_y = train_test_split(data_x, y, test_size=0.2)

f = open('2a.txt', 'w')

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
pyplot.title('Question 2.a')
pyplot.plot(nn.validation_scores_)
pyplot.savefig('2a.jpg', dpi = 100)
f.close()

with open('model_2a.pkl', 'wb') as output:
    pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)
