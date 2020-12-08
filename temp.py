import numpy as np
import pandas as pd
import cv2
import utils
import os


# def fn():       # 1.Get file names from directory
#     file_list = os.listdir(r"C:\Users\hunglt16\Workspace\DL\Project\FBD\SCUT-FBP5500_v2\Images")
#     return np.array(file_list)


# import data
(x_train, y_train), (x_test, y_test) = utils.get_data('SCUT-FBP5500_v2/')
# y_train_beauty = y_train[:, 0]
# y_train_race = y_train[:, 1]
# y_train_gender = y_train[:, 2]
# y_test_beauty = y_test[:, 0]
# y_test_race = y_test[:, 1]
# y_test_gender = y_test[:, 2]

# # Split data to train & test
# x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=0.2, random_state=6)

print(x_train.shape)
# print(y_train.shape)
# print(y_train_beauty.shape)
# print(y_train_race.shape)
# print(y_train_gender.shape)
print(y_train[:5])
