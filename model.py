from scipy.io import loadmat
# from skimage import io
# from skimage.transform import resize
from sklearn.model_selection import train_test_split
# import keras

import pandas as pd
import numpy as np
import os

from os.path import join

# import h5py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from IPython.display import clear_output

import tensorflow.keras as keras
# from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from keras.utils import plot_model

from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# model architecture
def my_model():
    pre_trained_model = ResNet50(weights='imagenet',
                                 include_top=False,
                                 input_shape=(224, 224, 3))
    for layer in pre_trained_model.layers:
        layer.trainable = False

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

    last_output = layers.GlobalAveragePooling2D()(last_output)

    # beauty branch
    beauty_branch = layers.Dense(64, activation='relu')(last_output)
    beauty_branch = layers.Dense(1, activation='relu', name='beauty_output')(beauty_branch)

    # race branch
    race_branch = layers.Dense(64, activation='relu')(last_output)
    race_branch = layers.Dense(1, activation='sigmoid', name='race_output')(race_branch)

    # gender branch
    gender_branch = layers.Dense(64, activation='relu')(last_output)
    gender_branch = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_branch)

    model = Model(inputs=pre_trained_model.input,
                  outputs=[beauty_branch, race_branch, gender_branch])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss={'beauty_output': 'mse',
                        'race_output': 'binary_crossentropy',
                        'gender_output': 'binary_crossentropy'},
                  loss_weights={'beauty_output': .6,
                                'race_output': .2,
                                'gender_output': .2})
    return model


def train_model(x_train, y_train_beauty, y_train_race, y_train_gender, filepath="weights_min_loss.hdf5"):
    model = my_model()
    model.summary()
    save_model = True
    if save_model:
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(x_train,
                        {'beauty_output': y_train_beauty,
                         'race_output': y_train_race,
                         'gender_output': y_train_gender},
                        epochs=15, batch_size=64,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_split=0.25)

    # Display curves of loss every epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.yscale("log")
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    plt.legend()
    fig_num = 0
    fig_name = "figures/Figure_loss"
    while os.path.exists('{}_{:d}.png'.format(fig_name, fig_num)):
        fig_num += 1
    plt.savefig('{}_{:d}.png'.format(fig_name, fig_num))


def predict_model(x_test, filepath):
    model = my_model()
    model.load_weights(filepath)
    y_pred = model.predict(x_test)
    y_pred_beauty = np.squeeze(y_pred[0])
    y_pred_race = np.round(np.squeeze(y_pred[1]))
    y_pred_gender = np.round(np.squeeze(y_pred[2]))
    return y_pred_beauty, y_pred_race, y_pred_gender


def demo(image, filepath):
    image = keras.preprocessing.image.load_img("images/" + image, target_size=(224, 224))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    y_pred_beauty, y_pred_race, y_pred_gender = predict_model(input_arr, filepath)
    print("Beauty score:", y_pred_beauty)
    print("Race: ", ("Asian" if y_pred_race == 0 else "Caucasian"))
    print("Gender: ", ("Female" if y_pred_gender == 0 else "Male"))







