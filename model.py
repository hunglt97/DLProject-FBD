from scipy.io import loadmat
from skimage import io
from skimage.transform import resize
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
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import plot_model

from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta

import utils


class PlotProgress(keras.callbacks.Callback): # Plot graph after each epoch

    def __init__(self, entity='loss'):
        self.entity = entity

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('{}'.format(self.entity)))
        self.val_losses.append(logs.get('val_{}'.format(self.entity)))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="{}".format(self.entity))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity))
        plt.legend()
        plt.show()


# import data
(x_train, y_train), (x_test, y_test) = utils.get_data('SCUT-FBP5500_v2/')
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

#normalize inputs between [0, 1]
x_train /= 255
x_test /= 255

# model architecture
# variable
batch_size = 256
epochs = 5
lr = 0.001
decay = lr/epochs

# Input
inputs = Input(shape=x_train[0].shape, name='main_input')


# load main-branch model
# main_branch = keras.models.load_model('path/to/location')(inputs)
def main_vgg():
    # the 1-st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_1')
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name='drop1_1')(pool1_1)

    # the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_3')(conv2_2)
    conv2_3 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)

    # the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), name='pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name='drop3_1')(pool3_1)

    # the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name='pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name='drop4_1')(pool4_1)

    # the 5-th block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_4')(conv5_3)
    conv5_4 = BatchNormalization()(conv5_4)
    pool5_1 = MaxPooling2D(pool_size=(2, 2), name='pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name='drop5_1')(pool5_1)

    # Flatten and output
    flatten = Flatten(name='flatten')(drop5_1)

    # create model
    main_branch_vgg = Model(inputs=conv1_1, outputs=flatten)

    return main_branch_vgg


main_branch = main_vgg()

# beauty branch
beauty_branch = Dense(1, activation='relu', name='beauty_output')(main_branch)

# race branch
race_branch = Dense(2, activation='softmax', name='race_output')(main_branch)

# gender branch
gender_branch = Dense(2, activation='softmax', name='gender_output')(main_branch)

# Model build
model = Model(inputs=inputs,
              outputs=[beauty_branch, race_branch, gender_branch])
model.summary()

opt = keras.optimizers.RMSprop(lr=1e-3)
model.compile(optimizer='rmsprop',
              loss={'age_output': 'mse', 'gender_output': 'sparse_categorical_crossentropy'},
              loss_weights={'age_output': .001, 'gender_output': 1.})

plot_progress = PlotProgress(entity='loss')

try:
    model.fit({'main_input': x_train},
              {'beauty_output': y_train_beauty,
               'race_output': y_train_race,
               'gender_output': y_train_gender},
              epochs=100, batch_size=128,
              verbose=1,
              callbacks=[plot_progress],
              validation_split=0.2,
             )
except KeyboardInterrupt:
    pass
