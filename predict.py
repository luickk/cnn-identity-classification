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
haar_cascade = 'data/haarcascade_frontalface_default.xml'

def main():
    input_shape = (150, 150, 3)
    model_directory = 'data/trainedModels'
    img = cv2.imread("data/data/val/elton_john/httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTEODAOTcxNjcMjczMjkzjpg.jpg")
    model_uuid = "f8ee03b4-e39a-11ea-979f-faffc2003e2a"

    faces_img_data, faces_data = detect_faces(haar_cascade , img)
    faces_img_data_color = img_resize(map_to_color(img, faces_data), 150, 150)
    for data in range(len(faces_img_data_color)):
        pred, class_dictionary = predict(img = faces_img_data_color[data], model_id = model_uuid, model_directory = model_directory, input_shape=input_shape)
        img_name = 'Face {}, Classes: {}'.format(str(pred),class_dictionary)
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(img_name, 600,600)
        cv2.imshow(img_name, faces_img_data_color[data])


def predict(img, model_id, model_directory, input_shape):
    class_dictionary_file = open(model_directory+'/'+'class_indices_file.txt', 'r')
    class_dictionary = class_dictionary_file.read()

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_dictionary)))
    model.add(Activation('softmax'))

    load_model(model_directory+'/model_'+model_id+'.h5')

    img_as_array = img_to_array(img)

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