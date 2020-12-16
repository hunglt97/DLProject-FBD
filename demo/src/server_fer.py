from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io
from flask import Flask, render_template, request
from tensorflow import keras
import cv2 as cv
from tensorflow.keras.models import model_from_json

app = flask.Flask(__name__)
model = None


def load_model():
    global classes_name
    global model
    global face_cascade
    global race
    global gender
    race = {
        '0': 'Asian',
        '1': 'Caucasian'
    }
    gender = {
        '0': 'Female',
        '1': 'Male'
    }
    classes_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    json_file = open('models/model_new.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models/weights_new.hdf5")

def prepare_image(image, target=(224,224)):

    gray = image
    width = image.shape[0]
    height = image.shape[1]
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    x, y, w, h = faces[0]
    ymin = np.max([0, int(np.round(y-h*0.05))])
    ymax = np.min([height, int(np.round(y+h*1.05))])
    xmin = np.max([0, int(np.round(x-w*0.05))])
    xmax = np.min([width, int(np.round(x+w*1.05))])
    print(faces[0])
    gray_face = gray[ymin:ymax, xmin:xmax]
    print('before border:',gray_face.shape)
    border_x = int(w*0.15)
    border_y = int(h*0.15)
    gray_face = cv.copyMakeBorder(
        gray_face,
        top=border_y,
        bottom=border_y,
        left=border_x,
        right=border_x,
        borderType=cv.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    print('after border:',gray_face.shape)
    gray_face = cv.resize(gray, (224,224))
    test_face = tf.keras.applications.resnet50.preprocess_input(gray_face[np.newaxis, :, :])
    return test_face

def get_predicted_image(image, image_name):
    gray = image
    width = image.shape[0]
    height = image.shape[1]
    faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=10, minSize=(30, 30),)
    for (x, y, w, h) in faces:
        #draw detected face
        face_rec = cv.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)

        ymin = np.max([0, int(np.round(y-h*0.1))])
        ymax = np.min([height, int(np.round(y+h*1.1))])
        xmin = np.max([0, int(np.round(x-w*0.1))])
        xmax = np.min([width, int(np.round(x+w*1.1))])
        print(faces[0])
        gray_face = gray[ymin:ymax, xmin:xmax]
        print('before border:',gray_face.shape)
        border_x = int(w*0.15)
        border_y = int(h*0.15)
        gray_face = cv.copyMakeBorder(
            gray_face,
            top=border_y,
            bottom=border_y,
            left=border_x,
            right=border_x,
            borderType=cv.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        print('after border:',gray_face.shape)
        gray_face = cv.resize(gray, (224,224))
        gray_face = gray_face[np.newaxis, :, :]
        test_face = tf.keras.applications.resnet50.preprocess_input(gray_face)
        preds = model.predict(test_face)
        r = int(np.around(np.squeeze(preds[1]))[()])
        g = int(np.around(np.squeeze(preds[2]))[()])
        score = str(np.around(np.squeeze(preds[0]), 2)[()])
        text = f'score: {score}\nrace: {race[str(r)]}\ngender: {gender[str(g)]}'
        dy = 20
        for i, line in enumerate(text.split('\n')):
            cv.putText(face_rec, line, (x, y + h + (i+1)*dy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    
    return cv.imwrite(f'static/images/{image_name}_score.jpg', image[:, :, ::-1])

@app.route('/')
def upload():
	return flask.render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		print(request)
		print(request.files)
		f = request.files.get('image')
		f.save(f.filename)
		return 'file uploaded successfully'


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image_name = flask.request.files["image"].filename
            image = np.array(Image.open(io.BytesIO(image)))
            print(image.shape)
            image_copy = image.copy()
            success = get_predicted_image(image_copy, image_name)
            result = {}
            result['success'] = success
            if not success:
                image = prepare_image(image, target=(224, 224))
                preds = model.predict(image)
                result['beauty_score'] = float(np.around(np.squeeze(preds[0]), 2)[()])
                r = int(np.around(np.squeeze(preds[1]))[()])
                g = int(np.around(np.squeeze(preds[2]))[()])
                result['race'] = race[str(r)]
                result['gender'] = gender[str(g)]
            else:
                pass
            print(result)
        return flask.jsonify(result)
if __name__ == "__main__":
	load_model()
	app.run(host="localhost", port=80,debug = False, threaded = False)

