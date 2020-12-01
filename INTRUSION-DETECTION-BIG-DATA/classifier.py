import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, TimeDistributed, GRU
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import os
# 加载数据
from tensorflow.python.keras.models import Sequential

train_all = np.load('data/encoded_train.npy')  # (175341, 32)
train_all_label = np.load('data/train_label.npy')  # (175341, 1)
test_all = np.load('data/encoded_test.npy')
test_all_label = np.load('data/test_label.npy')


# 利用TimesereisGenerator生成序列数据
time_steps = 8
batch_size = 128

# 先把训练集划分出一部分作为验证集
train = train_all[:(172032+time_steps), :]   # 4096 * 42 = 172032
train_label = train_all_label[:(172032+time_steps), :]
test = test_all[:(81920+time_steps), :]  # 4096 * 20 = 81920
test_label = test_all_label[:(81920+time_steps), :]
# val_data = train_all[int(len(train_all)* 0.7):, :]
# val_label = train_all_label[int(len(train_all)* 0.7):, :]
# print(train.shape[0])
# print(val_data.shape[0])
# 数据集生成器
train_label_ = np.insert(train_label, 0, 0, axis=0)
test_label_ = np.insert(test_label, 0, 0, axis=0)
# val_label_ = np.insert(val_label, 0, 0)
train_generator = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
# val_generator = TimeseriesGenerator(val_data, val_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)

# 构造模型
# input_traffic = Input((time_steps, 32))
input_traffic = Input(shape=(time_steps, 196))

'''GRU1=Bidirectional(GRU(units=24, activation='tanh',
                           return_sequences=True, recurrent_dropout=0.1))(input_traffic)
GRU_drop1 = Dropout(0.5)(GRU1)

GRU2=Bidirectional(GRU(units=12, activation='tanh',
                           return_sequences=False, recurrent_dropout=0.1))(GRU_drop1)
GRU_drop2 = Dropout(0.5)(GRU2)'''
Dense1=Dense(128, activation='relu', use_bias=True)(input_traffic)
dense_drop1 = Dropout(0.5)(Dense1)
Dense2=Dense(64, activation='relu', use_bias=True)(dense_drop1)
dense_drop2 = Dropout(0.5)(Dense2)
Dense3=Dense(32, activation='relu', use_bias=True)(dense_drop2)
dense_drop3 = Dropout(0.5)(Dense3)
Dense4=Dense(1, 'sigmoid', use_bias=True)(dense_drop3)# The number of neurons is equal to the number of classes
model = Model(input_traffic, Dense4)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
reduc_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='max', factor=0.2, min_delta=0.0001)

# 拟合及预测
history = model.fit_generator(train_generator, epochs=40, verbose=2, steps_per_epoch=100,
                                   callbacks=reduc_lr,
                                   validation_data=test_generator, shuffle=0,validation_steps=50)


train_probabilities = model.predict(train_generator, verbose=1)

train_pred = train_probabilities > 0.5
train_label_original = train_label_[(time_steps-1):-2, :]

test_probabilities = model.predict(test_generator, verbose=1)
test_pred = test_probabilities > 0.5
test_label_original = test_label_[(time_steps-1):-2, ]
np.save('data/plot_prediction.npy', test_pred)
np.save('data/plot_original.npy', test_label_original)
# tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
from sklearn.metrics import confusion_matrix, classification_report
'''print('Trainset Confusion Matrix')
print(confusion_matrix(train_label_original, train_pred))
print('Testset Confusion Matrix')
print(confusion_matrix(test_label_original, test_pred))
print('Classification Report')

print(classification_report(test_label_original, test_pred))
'''