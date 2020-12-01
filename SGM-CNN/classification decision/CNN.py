
import os
# load dataset
from tensorflow.keras.layers import Input,Flatten,Conv1D, MaxPooling1D, Dropout, BatchNormalization, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Nadam
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import time

x_train = np.load('F:/数据挖掘/SGM-CNN/新建文件夹/SGM_data_train.npy')
y_train = np.load('F:/数据挖掘/SGM-CNN/新建文件夹/SGM_label10_train.npy')

x_test = np.load('F:/数据挖掘/SGM-CNN/新建文件夹/test_select_12_data.npy')
y_test = np.load('F:/数据挖掘/SGM-CNN/新建文件夹/test_select_12_label_10.npy')

x_val = np.load('F:/数据挖掘/SGM-CNN/新建文件夹/validation_select_12_data.npy')
y_val = np.load('F:/数据挖掘/SGM-CNN/新建文件夹/validation_select_12_label_10.npy')


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape)

x_train = np.expand_dims(x_train,2)
x_test = np.expand_dims(x_test,2)
x_val = np.expand_dims(x_val,2)
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
    x = Dense(10, activation='softmax')(x)  # UNSW-NB15 is 2 and 10,CICIDS2017 is 15
    model = Model(inputs=input_singal, outputs=x)

    return model

model =  build_model()
model.summary()

time_start = time.time()

reduce_lr = keras.callbacks.ReduceLROnPlateau(moniter='val_loss',
                                              factor=0.1,
                                              patience=10)
nadam = Nadam(lr=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss = "categorical_crossentropy",optimizer = "nadam", metrics = ["accuracy"])

save_dir = os.path.join(os.getcwd(), 'model_save')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filename="model_{epoch:02d}.hdf5"
ckpt_callback=tf.keras.callbacks.ModelCheckpoint(
                os.path.join(save_dir, filename),monitor='val_acc'
                , verbose=1, save_best_only=False)
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(tf.train.latest_checkpoint("F:/数据挖掘/SGM-CNN/classification decision/model_save"))
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(x_val, y_val),
                    callbacks=[reduce_lr,ckpt_callback])
time_end = time.time()
train_time = time_end - time_start
print("train_time:",train_time)

scores = model.evaluate(x_test, y_test)
print("test_loss = ", scores[0],"test_accuracy = ", scores[1])

model.save('F:/数据挖掘/SGM-CNN/model_out')#save model

# test model

import time
time_start = time.time()

y_pred_onehot  = model.predict(x_test)  #返回的是在类别上的概率分布.It returns the probability distribution on the category
y_pred_label=np.argmax(y_pred_onehot,axis=1)#概率最大的类别就是预测类别.The category with the highest probability is the prediction category

time_end = time.time()
test_time = time_end - time_start
print("test_time:",test_time)

# np.savetxt("E:/IDS/cicdata/GMM+SMOTE_77/2ci/CNN_pred_15.txt",y_pred_label)


y_true_onehot=y_test
y_true_label=np.argmax(y_true_onehot,axis=1)
# np.savetxt("E:/IDS/cicdata/GMM+SMOTE_77/2ci/CNN_true_15.txt",y_true_label)

labels = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']  #class name

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
#plt.savefig('/home/hll/IDS/alldata/cm/confusion_matrix.png', format='png')
plt.show()


print(cm)  #Confusion matrix

# multi-class evaluation indicators

from sklearn import metrics
from sklearn.metrics import classification_report

target_names = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']
print(classification_report(y_true,y_pred,target_names=target_names))


acc = metrics.accuracy_score(y_true,y_pred)
f1 = metrics.f1_score(y_true, y_pred,average='weighted')
pre = metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='weighted')  #DR
recall = metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

print("acc:",acc)
print("pre:",pre)
print("DR=recall:",recall)
print("f1:",f1)

