import numpy as np
from pandas import DataFrame
num=0
labelsAndPredictions=np.array([[[0,1],[1,2],[2,3]],[[1,2],[2,3],[3,4]]])
si = labelsAndPredictions.shape[1]
label_t = labelsAndPredictions[0]
label_p = labelsAndPredictions[1]
for i in range(si):
    if label_t[i] == label_p[i]:
        num += 1
print(num)