import numpy as np
import time

from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# Load the dataset and labels
from tensorflow.python.layers.core import dense

x = np.load('data/encoded_train.npy')
y = np.load('data/train_label.npy')
a = [[1], [2], [3]]
b = [1, 2, 3]
y = y.reshape(y.shape[0])
# Calculate the rank of each feature
R = []
for h in range(x.shape[1]):
    kmeans = KMeans(init='k-means++', n_clusters=np.unique(y).shape[0], n_init=10)
    ff = kmeans.fit_predict(x[:, h].reshape(-1, 1))
    r = metrics.homogeneity_score(y, ff)  # Use the homogeneity score as a rank of the feature
    R.append(r)

# Arrange feature accroding to thier ranks
Rnk = np.argsort(np.array(R))

# Initiate the cross-validation splitter
kfolds = StratifiedKFold(n_splits=5, shuffle=True)

# Per each set of ranks, use cross-validation to calculate accuracy.
smr = []
for j in range(Rnk.shape[0]):
    fd = x[:, Rnk[j:]]
    pp = 0
    lpa = np.zeros((0, 2))
    for train, test in kfolds.split(fd, y):
        print(fd.shape)
        print(train.shape)
        train_t = map(lambda x: (int(float(x[-1])), dense(x[:-1])), np.hstack((fd[train], y[train].reshape(-1, 1))))
        test_t = map(lambda x: (int(float(x[-1])), dense(x[:-1])), np.hstack((fd[test], y[test].reshape(-1, 1))))
        model = Sequential()
        model.add(Dense(units=128, input_shape=fd.shape, activation='relu', use_bias=True))
        model.add(Dropout(0.5))
        model.add(Dense(units=64, activation='relu', use_bias=True))
        model.add(Dropout(0.5))
        model.add(Dense(units=32, activation='relu', use_bias=True))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid',
                        use_bias=True))  # The number of neurons is equal to the number of classes
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(train_t, epochs=40, verbose=2)
        model.predict(test_t, verbose=1)
        ts = np.array(map(lambda x: x[0], ff.select('prediction').collect())).reshape(-1, 1)
        pp = pp + metrics.accuracy_score(y[test].reshape(-1, 1), (ts >= 0.5).astype(int))
        lpa = np.vstack((lpa, np.hstack((y[test].reshape(-1, 1), ts))))
    pp = pp / kfolds.n_splits
    np.savetxt('F%d.csv' % j, lpa, delimiter=',')
'''    smr.append(
        [j, pp, et * 1000000 / x.shape[0]])  # Calculate the time required to predict a label per each object in uS.'''
