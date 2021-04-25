import os
import numpy as np
# load dataset
from tensorflow.keras.layers import Input, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Nadam
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import time

x_train = np.load('../dataset/SGM-CNN/新建文件夹/train_select_12_data.npy')
y_train = np.load('../dataset/SGM-CNN/新建文件夹/train_select_12_label_2.npy')

x_test = np.load('../dataset/SGM-CNN/新建文件夹/test_select_12_data.npy')
y_test = np.load('../dataset/SGM-CNN/新建文件夹/test_select_12_label_2.npy')

x_val = np.load('../dataset/SGM-CNN/新建文件夹/validation_select_12_data.npy')
y_val = np.load('../dataset/SGM-CNN/新建文件夹/validation_select_12_label_2.npy')

x_train = np.expand_dims(x_train, 2)
x_test = np.expand_dims(x_test, 2)
x_val = np.expand_dims(x_val, 2)
# label one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# train model


np.random.seed(4)

n_obs, feature, depth = x_train.shape
batch_size = 256


def build_model():
    input_singal = Input(shape=(feature, depth))
    x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(input_singal)
    x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)  # UNSW-NB15 is 2 and 10,CICIDS2017 is 15
    model = Model(inputs=input_singal, outputs=x)
    return model


model = build_model()
model.summary()

time_start = time.time()

reduce_lr = keras.callbacks.ReduceLROnPlateau(moniter='val_loss',
                                              factor=0.1,
                                              patience=10)
nadam = Nadam(lr=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

save_dir = os.path.join(os.getcwd(), 'model_save')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filename = "model_{epoch:02d}.hdf5"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(save_dir, filename), monitor='val_acc'
    , verbose=1, save_best_only=False)
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(tf.train.latest_checkpoint("../dataset/SGM-CNN/classification decision/model_save"))
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(x_val, y_val),
                    callbacks=[reduce_lr, ckpt_callback])
time_end = time.time()
train_time = time_end - time_start
print("train_time:", train_time)

scores = model.evaluate(x_test, y_test)
print("test_loss = ", scores[0], "test_accuracy = ", scores[1])

model.save('../dataset/SGM-CNN/model_out')  # save model
# test model


time_start = time.time()

y_pred_onehot = model.predict(x_test)  # 返回的是在类别上的概率分布.It returns the probability distribution on the category
y_pred_label = np.argmax(y_pred_onehot,axis=1)  # 概率最大的类别就是预测类别.The category with the highest probability is the prediction category

time_end = time.time()
test_time = time_end - time_start
print("test_time:", test_time)

y_true_onehot = y_test
y_true_label = np.argmax(y_true_onehot, axis=1)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

labels = ['Normal', 'Attack']  # class name

y_true = y_true_label
y_pred  = y_pred_label

tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(15, 13), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.001:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=13, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='MLP_12_10_ROS Normalized confusion matrix')
plt.show()


print(cm)  #Confusion matrix

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('MLP_12_2_CNN accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('MLP_12_2_CNN loss')
plt.legend()
plt.show()


# binary-class evaluation indicators

TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

acc = (TP + TN) / (TP + TN + FP + FN)
print("acc:", acc)

DR = TP / (TP + FN)
print("DR:", DR)

FPR = FP / (FP + TN)  # FAR
print("FPR:", FPR)

recall = TP / (TP + FN)
print("recall：", recall)

precision = TP / (TP + FP)
print("precision:", precision)

f1 = (2 * precision * recall) / (precision + recall)
print("f1:", f1)
