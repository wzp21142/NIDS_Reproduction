# SMOTE all minority classes to (number of data set samples / classes)

from collections import Counter
import numpy as np
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE
import time

data = np.load('../dataset/SGM-CNN/新建文件夹/train_select_12_data.npy')
label = np.load('../dataset/SGM-CNN/新建文件夹/train_select_12_label_10.npy')

X = np.array(data)
b = np.array(label)
bb = b.reshape(b.shape[0], )
y10 = np.int32(bb)

print(X.shape)

sorted(Counter(y10).items())


time_start = time.time()
a = 177803
# [X[0].size/10]

smo = SMOTE(sampling_strategy={1:a, 2: a, 3: a, 4: a, 5: a, 6: a, 7: a, 8: a, 9: a}, random_state=42)

X_smo, y_smo = smo.fit_sample(X, y10)
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
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
c7 = []
c8 = []
c9 = []
s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = s9 = 0

for i in range(data0.shape[0]):
    if label_pred[i] == 0:
        c0.append(data0[i])
        s0 = s0 + 1
    elif label_pred[i] == 1:
        c1.append(data0[i])
        s1 = s1 + 1
    elif label_pred[i] == 2:
        c2.append(data0[i])
        s2 = s2 + 1
    elif label_pred[i] == 3:
        c3.append(data0[i])
        s3 = s3 + 1
    elif label_pred[i] == 4:
        c4.append(data0[i])
        s4 = s4 + 1
    elif label_pred[i] == 5:
        c5.append(data0[i])
        s5 = s5 + 1
    elif label_pred[i] == 6:
        c6.append(data0[i])
        s6 = s6 + 1
    elif label_pred[i] == 7:
        c7.append(data0[i])
        s7 = s7 + 1
    elif label_pred[i] == 8:
        c8.append(data0[i])
        s8 = s8 + 1
    elif label_pred[i] == 9:
        c9.append(data0[i])
        s9 = s9 + 1

a = 17780
# [a/10]

del c1[a:len(c1)]
del c2[a:len(c2)]
del c3[a:len(c3)]
del c4[a:len(c4)]
del c5[a:len(c5)]
del c6[a:len(c6)]
del c7[a:len(c7)]
del c8[a:len(c8)]
del c9[a:len(c9)]

c00 = np.array(c0)
c11 = np.array(c1)
c22 = np.array(c2)
c33 = np.array(c3)
c44 = np.array(c4)
c55 = np.array(c5)
c66 = np.array(c6)
c77 = np.array(c7)
c88 = np.array(c8)
c99 = np.array(c9)

q = np.concatenate((c00, c11, c22, c33, c44, c55, c66, c77, c88, c99), axis=0)
q.shape
label_zc = np.zeros((q.shape[0],), dtype=int)
label_zc.shape

data_end = np.concatenate((q, data1), axis=0)
label_end = np.concatenate((label_zc, label1), axis=0)

sorted(Counter(label_end).items())

label_end = label_end.reshape(label_end.shape[0], 1)

np.save("../dataset/SGM-CNN/新建文件夹/SGM_data_train.npy", data_end)
np.save("../dataset/SGM-CNN/新建文件夹/SGM_label10_train.npy", label_end)