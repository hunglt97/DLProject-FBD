import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
import numpy as np
import model as fbdnet
import utils1

if __name__ == '__main__':
    # import data
    (x_train, y_train), (x_test, y_test) = utils1.get_data('SCUT-FBP5500_v2/')
    y_train_beauty = y_train[:, 0]
    y_train_race = y_train[:, 1]
    y_train_gender = y_train[:, 2]
    y_test_beauty = y_test[:, 0]
    y_test_race = y_test[:, 1]
    y_test_gender = y_test[:, 2]

    x_train = x_train.reshape(3300, 224, 224, 3)
    x_test = x_test.reshape(2200, 224, 224, 3)
    x_train = keras.applications.resnet50.preprocess_input(x_train)
    x_test = keras.applications.resnet50.preprocess_input(x_test)

    filepath = "weights.hdf5"
    # model = fbdnet.my_model()
    # fbdnet.train_model(x_train, y_train_beauty, y_train_race, y_train_gender, filepath)

    y_pred_beauty, y_pred_race, y_pred_gender = fbdnet.predict_model(x_test, filepath)

    cov = np.cov(y_pred_beauty, y_test_beauty)
    print('pc =', cov/(np.std(y_pred_beauty)*np.std(y_test_beauty)))
    print('mae =', np.mean(np.abs(y_pred_beauty-y_test_beauty)))
    print('rmse =', np.sqrt(np.mean(np.square(y_pred_beauty-y_test_beauty))))
    print('race acc', accuracy_score(y_test_race, y_pred_race))
    print('gender acc', accuracy_score(y_test_gender, y_pred_gender))

    for i in range(2, 200, 40):
        test_img = x_test[i]
        test_img = test_img[np.newaxis, :, :, :]
        y_pred_beauty_1, y_pred_race_1, y_pred_gender_1 = fbdnet.predict_model(test_img, filepath)
        print(y_test_beauty[i], y_pred_beauty_1)
        print(y_test_race[i], y_pred_race_1)
        print(y_test_gender[i], y_pred_gender_1)

    fbdnet.demo('test1.jpg', filepath)
    fbdnet.demo('test2.jpg', filepath)
    fbdnet.demo('test3.jpg', filepath)
    fbdnet.demo('test4.jpg', filepath)






