import numpy as np
import cv2
import os
import sys
import model as fbdnet

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
name = "images/bp"
image = cv2.imread(name + '.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=10,
    minSize=(30, 30),
    # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

filepath = "weights.hdf5"
id = 0
for (x, y, w, h) in faces:
    cropped = image[int(np.round(y-h*0.2)):y+int(np.round(h*1.2)), int(np.round(x-w*0.2)):x+int(np.round(w*1.2))]
    cropped = cv2.resize(cropped, (180, 180))
    # row, col = cropped.shape[:2]
    # bottom = cropped[row - 2:row, 0:col]
    # mean = cv2.mean(bottom)[0]

    border_size = 22
    border = cv2.copyMakeBorder(
        cropped,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    while os.path.exists(name + str(id) + ".jpg"):
        id += 1
    cv2.imwrite(name + str(id) + ".jpg", border)
    face_rec = cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
    border1 = np.array([border])
    y_pred_beauty, y_pred_race, y_pred_gender = fbdnet.predict_model(border1, filepath)
    race = "Race: " + ("Asian" if y_pred_race == 0 else "Caucasian")
    gender = "Gender: " + ("Female" if y_pred_gender == 0 else "Male")
    text = str(np.round(y_pred_beauty,2)) + "\n" + str(race) + "\n" + str(gender)
    dy = 20
    for i, line in enumerate(text.split('\n')):
        cv2.putText(face_rec, line, (x, y + h + (i+1)*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

cv2.imwrite(name + "_demo" + str(id) + ".jpg", image)




# cv2.imshow('image', image)
# cv2.imshow('bottom', bottom)
# cv2.imshow('border', border)
# cv2.waitKey(0)
# cv2.destroyAllWindows()