import time

import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.ops.clustering_ops import KMeans
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
R = []
x = np.load('Bx.npy')
y = np.load('By.npy')
for h in range(x.shape[1]):
    kmeans = KMeans(init='k-means++', n_clusters=np.unique(y).shape[0])
    # The number of clusters is set to the number of classes in the dataset
    ff = kmeans.fit_predict(x[:, h].reshape(-1, 1))
    r = metrics.homogeneity_score(y, ff)  # Use the homogeneity score as a rank of the feature
    R.append(r)
Rnk = np.argsort(np.array(R))
# Initiate the cross-validation splitter
kfolds = StratifiedKFold(n_splits=5, shuffle=True)
est = tf.estimator.BoostedTreeClassifier()
# Per each set of ranks, use cross-validation to calculate accuracy.

smr = []
et = 0
for j in range(Rnk.shape[0]):
    fd = x[:, Rnk[j:]]
    pp = 0
    lpa = np.zeros((0, 2))
    for train, test in kfolds.split(fd, y):
        dff = map(lambda x: (int(float(x[-1])), Vectors.dense(x[:-1])), np.hstack((fd[train], y[train].reshape(-1, 1))))
        TrD = spark.createDataFrame(dff, schema=["label", "features"]).rdd.map(
            lambda row: LabeledPoint(row.label, MLLibVectors.fromML(row.features)))
        dff = map(lambda x: (int(float(x[-1])), Vectors.dense(x[:-1])), np.hstack((fd[test], y[test].reshape(-1, 1))))
        TsD = spark.createDataFrame(dff, schema=["label", "features"]).rdd.map(
            lambda row: LabeledPoint(row.label, MLLibVectors.fromML(row.features)))
        model = GradientBoostingClassifier.trainClagssifier(TrD, categoricalFeaturesInfo={})
        predictions = model.predict(TsD.map(lambda x: x.features))
        st = time.time()
        labelsAndPredictions = TsD.map(lambda lp: lp.label).zip(predictions)
        lpa = np.vstack((lpa, labelsAndPredictions.toDF().toPandas()))
        et += time.time() - st
        acc = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(TsD.count())
        pp = pp + acc
    pp = pp / kfolds.n_splits
    np.savetxt('F%d.csv' % j, lpa, delimiter=',')
    smr.append(
        [j, pp, et * 1000000 / x.shape[0]])  # Calculate the time required to predict a label per each object in uS.
