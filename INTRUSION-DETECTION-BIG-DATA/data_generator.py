from data_processing import load_data
from build_model import build_SAE
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Reshape, Dropout
import os
# load data
print("Load data...")
train, train_label, test, test_label = load_data()
print("train shape: ", train.shape)
train_label = train_label.reshape((-1, 1))
test_label = test_label.reshape((-1, 1))
print("train_label shape: ", train_label.shape)



np.save('data/encoded_train.npy', train)
np.save('data/train_label.npy', train_label)
np.save('data/encoded_test.npy', test)
np.save('data/test_label.npy', test_label)