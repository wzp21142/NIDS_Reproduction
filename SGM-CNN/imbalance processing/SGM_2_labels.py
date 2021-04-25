# SMOTE all minority classes to (number of data set samples / classes)

from collections import Counter
import numpy as np

data = np.load('../dataset/SGM-CNN/新建文件夹/train_select_12_data.npy')
label = np.load('../dataset/SGM-CNN/新建文件夹/train_select_12_label_2.npy')

X = np.array(data)
b = np.array(label)
bb = b.reshape(b.shape[0], )
y2 = np.int32(bb)

print(X.shape)

sorted(Counter(y2).items())
from imblearn.over_sampling import SMOTE
import time

time_start = time.time()
a = 889015
# [X[0].size/2]

smo = SMOTE(sampling_strategy={1:a}, random_state=42)

X_smo, y_smo = smo.fit_sample(X, y2)
print(sorted(Counter(y_smo).items()))

time_end = time.time()
time = time_end - time_start
print("time:", time)

print(X_smo.shape[0])

# Extract Majority class of data

list0 = []
list1 = []
list2 = []

for i in range(X_smo.shape[0]):
    if y_smo[i] == 0:
        list0.append(X_smo[i])  # 正常流量
    else:
        list1.append(X_smo[i])
        list2.append(y_smo[i])

data0 = np.array(list0)
data1 = np.array(list1)
label1 = np.array(list2)

label11 = label1.reshape(label1.shape[0], )

print("Normal class data shape：", data0.shape)
print("Attack class data shape：", data1.shape)
print("Attack class label shape：", label11.shape)

# Cluster majority data into  C (total number of classes)

from sklearn.mixture import GaussianMixture
import time

time_start = time.time()

estimator = GaussianMixture(n_components=10)
estimator.fit(data0)

time_end = time.time()
time = time_end - time_start
print("time:", time)

label_pred = estimator.predict(data0)

sorted(Counter(label_pred).items())

# Select a certain amount of data from each cluster to form a new majority data


c0 = []
c1 = []
s0 = s1 = 0

for i in range(data0.shape[0]):
    if label_pred[i] == 0:
        c0.append(data0[i])
        s0 = s0 + 1
    elif label_pred[i] == 1:
        c1.append(data0[i])
        s1 = s1 + 1
a = 444507
# [a/2]
del c1[a:len(c1)]
c00 = np.array(c0)
c11 = np.array(c1)

q = np.concatenate((c00, c11), axis=0)

label_zc = np.zeros((q.shape[0],), dtype=int)

data_end = np.concatenate((q, data1), axis=0)
label_end = np.concatenate((label_zc, label1), axis=0)


sorted(Counter(label_end).items())

label_end = label_end.reshape(label_end.shape[0], 1)

np.save("../dataset/SGM-CNN/新建文件夹/SGM_data_train.npy", data_end)
np.save("../dataset/SGM-CNN/新建文件夹/SGM_label2_train.npy", label_end)