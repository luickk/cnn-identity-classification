import tensorflow as tf
import numpy as np
import sys
import cv2

from cv import detect_faces

from cnn_tens import train_net as tens_train_net
from cnn_tens import label_pic as tens_label_pic

from cnn_keras import train_net as keras_train_net
from cnn_keras import label_pic as keras_label_pic

def main():
    imagePath = sys.argv[1]
    img = cv2.imread(imagePath)
    #img = cv2.VideoCapture(0).read()[1]
    haar_cascade = "cv/cascades/haarcascade_frontalface_default.xml"

    faces_img_data, faces_data = detect_faces.detect_faces(haar_cascade , img)
    faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 200, 200)

    #for data in range(len(faces_img_data_color)):
    #    cv2.imshow("face {}".format(data), faces_img_data_color[data])

    net_keras = keras_train_net.seq_net()
    net_keras.load_model(img_width = 150, img_height = 150, train_data_dir = 'img_data/train', validation_data_dir = 'img_data/validation', model_directory = 'cnn_keras/models')
    model, class_dictionary, model_directory = net_keras.retrain(epochs = 50, batch_size = 16, nb_train_samples = 2000, nb_validation_samples = 800)

    pred, class_dictionary = keras_label_pic.label_pic(img_path = imagePath, img_width = 150, img_height = 150, model_id = model_directory, model_directory = 'cnn_keras/models')

    print(pred)
    print(class_dictionary)


if __name__ == "__main__":
    main()
