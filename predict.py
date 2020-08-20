from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image as pil

import cv2
import numpy as np

# Defineing haarcascade for face detection
haar_cascade = 'cv/cascades/haarcascade_frontalface_default.xml'

def main():
    img = cv2.imread("")
    model_uuid = ""

    faces_img_data, faces_data = detect_faces.detect_faces(haar_cascade , img)
    faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 150, 150)

    for data in range(len(faces_img_data_color)):
        pred, class_dictionary = predict(img = faces_img_data_color[data], model_id = model_uuid, model_directory = 'pretrained_models')
        img_name = 'Face {}, Classes: {}'.format(str(pred),class_dictionary)
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(img_name, 600,600)
        cv2.imshow(img_name, faces_img_data_color[data])

def predictImg(img_path, img_width, img_height, model_id, model_directory = 'cnn_keras/models'):
    model = load_model(model_directory+'/'+model_id+'/'+'model.h5')
    model.load_weights(model_directory+'/'+model_id+'/'+'model_weights.h5')

    img_pil = pil.open(img_path)
    img_pil = img_pil.resize((img_width, img_height))
    img_as_array = img_to_array(img_pil)
    class_dictionary_file = open(model_directory+'/'+model_id+'/'+'class_indices_file.txt', 'r')
    class_dictionary = class_dictionary_file.read()
    return model.predict_classes(img_as_array), class_dictionary

def predict(img, model_id, model_directory = 'cnn_keras/models'):
    model = load_model(model_directory+'/'+model_id+'/'+'model.h5')
    model.load_weights(model_directory+'/'+model_id+'/'+'model_weights.h5')

    img_as_array = img_to_array(img)
    class_dictionary_file = open(model_directory+'/'+model_id+'/'+'class_indices_file.txt', 'r')
    class_dictionary = class_dictionary_file.read()
    return model.predict_classes(img_as_array), class_dictionary

def img_to_array(img):
    x = np.asarray(img, dtype=K.floatx())
    #expand dim for Keras net
    x = np.expand_dims(x / 255, axis=0)
    return x

def detect_faces(cascPath, img):
    faces_data = []
    faces_img_data = []
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cropped = gray[y-200:y+h+400, x-200:x+w+400]
        faces_img_data.append(cropped)
        faces_data.append([x-200, y-200, w+400, h+400])

    return faces_img_data, faces_data

def map_to_color(org_image, face_cord):
    color_face_img = []
    for (x, y, w, h) in face_cord:
        color_face_img.append(org_image[y:y+h, x:x+w])
    return color_face_img

def img_resize(faces, x, y):
    resize_face_img = []
    for data in range(len(faces)):
        resize_face_img.append(cv2.resize(faces[data], (x, y)))
    return resize_face_img



if __name__ == "__main__":
    main()