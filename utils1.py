import numpy as np
import pandas as pd
# import cv2
# import os
import tensorflow as tf
from tensorflow import keras


def get_auxiliary(data):
    features = np.zeros((len(data), 2))
    for i in range(len(data)):
        if (data[i][0] == 'A') & (data[i][1] == 'F'):
            features[i, :] = [0, 0]
        elif (data[i][0] == 'A') & (data[i][1] == 'M'):
            features[i, :] = [0, 1]
        elif (data[i][0] == 'C') & (data[i][1] == 'F'):
            features[i, :] = [1, 0]
        elif (data[i][0] == 'C') & (data[i][1] == 'M'):
            features[i, :] = [1, 1]
    return features


def get_data(data_path='SCUT-FBP5500_v2'):
    data_train = pd.read_csv(f'{data_path}/train_test_files/split_of_60%training and 40%testing/train.txt', sep=" ",
                             header=None)
    data_test = pd.read_csv(f'{data_path}/train_test_files/split_of_60%training and 40%testing/test.txt', sep=" ",
                            header=None)
    img_path = f'{data_path}/Images'
    x_test = []
    for i in range(len(data_test[0])):
        image = tf.keras.preprocessing.image.load_img(f'{img_path}/{data_test[0][i]}', target_size=(224, 224))
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        x_test.append(input_arr)
    x_test = np.array(x_test, dtype=float)
    names_test = data_test[data_test.columns[0]].tolist()
    aux_test = get_auxiliary(names_test)
    y_test = np.concatenate((np.array(data_test[1]).reshape(-1, 1), aux_test), axis=1)

    x_train = []
    for i in range(len(data_train[0])):
        image = tf.keras.preprocessing.image.load_img(f'{img_path}/{data_train[0][i]}', target_size=(224, 224))
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        x_train.append(input_arr)
    x_train = np.array(x_train, dtype=float)
    names_train = data_train[data_train.columns[0]].tolist()
    aux_train = get_auxiliary(names_train)
    y_train = np.concatenate((np.array(data_train[1]).reshape(-1, 1), aux_train), axis=1)

    return (x_train, y_train), (x_test, y_test)