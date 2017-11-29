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
    faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 150, 150)

    net_keras = keras_train_net.seq_net()
    net_keras.load_model(img_width = 150, img_height = 150, train_data_dir = 'img_data/dog_cat/train', validation_data_dir = 'img_data/dog_cat/validation', model_directory = 'cnn_keras/models')
    model, class_dictionary, model_uuid = net_keras.retrain(epochs = 50, batch_size = 16, nb_train_samples = 2000, nb_validation_samples = 800)



    for data in range(len(faces_img_data_color)):
        pred, class_dictionary = keras_label_pic.label_pic_no_path(img = faces_img_data_color[data], model_id = model_uuid, model_directory = 'cnn_keras/models')
        img_name = 'face {}, {}, Classes: {}'.format('Face ' + str(pred),data,class_dictionary)
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(img_name, 600,600)
        cv2.imshow(img_name, faces_img_data_color[data])

    print(class_dictionary)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
