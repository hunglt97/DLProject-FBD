from scipy.io import loadmat
# from skimage import io
# from skimage.transform import resize
from sklearn.model_selection import train_test_split
import keras

import pandas as pd
import numpy as np

from os.path import join

# import h5py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from IPython.display import clear_output

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from keras.utils import plot_model

from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta

import utils1


# import data
(x_train, y_train), (x_test, y_test) = utils1.get_data('SCUT-FBP5500_v2/')
y_train_beauty = y_train[:, 0]
y_train_race = y_train[:, 1]
y_train_gender = y_train[:, 2]
y_test_beauty = y_test[:, 0]
y_test_race = y_test[:, 1]
y_test_gender = y_test[:, 2]

# # Split data to train & test
# x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=0.2, random_state=6)

print(x_train.shape)
print(y_train.shape)
print(y_train_beauty.shape)
print(y_train_race.shape)
print(y_train_gender.shape)
print(y_train[:5])

x_train = x_train.reshape(3300, 224, 224, 3)
x_test = x_test.reshape(2200, 224, 224, 3)
# normalize inputs between [0, 1]
x_train = keras.applications.resnet50.preprocess_input(x_train)
x_test = keras.applications.resnet50.preprocess_input(x_test)

# model architecture
pre_trained_model = ResNet50(weights='imagenet',
                             include_top=False,
                             input_shape=(224, 224, 3))
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()
# # variable
# batch_size = 256
# epochs = 5
# lr = 0.001
# decay = lr/epochs

for i in range(1, 4):
    for j in range(1, 4):
        pre_trained_model.get_layer(f'conv5_block{i}_{j}_bn').trainable = True
        pre_trained_model.get_layer(f'conv5_block{i}_{j}_conv').trainable = True

for i in range(1, 7):
    for j in range(1, 4):
        pre_trained_model.get_layer(f'conv4_block{i}_{j}_bn').trainable = True
        pre_trained_model.get_layer(f'conv4_block{i}_{j}_conv').trainable = True

last_layer = pre_trained_model.get_layer('conv5_block3_out')
last_output = last_layer.output

# beauty branch
beauty_branch = Dense(1, activation='relu', name='beauty_output')(last_output)

# race branch
race_branch = Dense(2, activation='softmax', name='race_output')(last_output)

# gender branch
gender_branch = Dense(2, activation='softmax', name='gender_output')(last_output)

model = keras.Model(inputs=pre_trained_model.input,
                    outputs=[beauty_branch, race_branch, gender_branch])
model.summary()


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


from keras.callbacks import ModelCheckpoint

filepath = "weights_min_loss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

opt = keras.optimizers.RMSprop(lr=1e-3)
model.compile(optimizer='rmsprop',
              loss={'beauty_output': 'mse',
                    'race_output': 'sparse_categorical_crossentropy',
                    'gender_output': 'sparse_categorical_crossentropy'},
              loss_weights={'beauty_output': .001,
                            'race_output': 1.,
                            'gender_output': 1.})

# train_datagen = keras.preprocessing.image.ImageDataGenerator()
# test_datagen = keras.preprocessing.image.ImageDataGenerator()
# train_generator = train_datagen.flow(x_train, y_train, )
# test_generator = train_datagen.flow(x_test, y_test, batch_size=32)

history = model.fit({'main_input': x_train},
                    {'beauty_output': y_train_beauty,
                     'race_output': y_train_race,
                     'gender_output': y_train_gender},
                    epochs=20, batch_size=32,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_split=0.2)





