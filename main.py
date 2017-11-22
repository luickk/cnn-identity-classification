import tensorflow as tf
import numpy as np
import sys
import cv2

from cv import detect_faces

from cnn_tens import train_net as tens_train_net
from cnn_tens import label_pic as tens_label_pic

from cnn_keras import train_net as keras_train_net
from cnn_keras import train_net_VGG16 as keras_train_net_v2

def main():
    imagePath = sys.argv[1]
    img = cv2.imread(imagePath)
    #img = cv2.VideoCapture(0).read()[1]
    haar_cascade = "cv/cascades/haarcascade_frontalface_default.xml"

    faces_img_data, faces_data = detect_faces.detect_faces(haar_cascade , img)
    faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 200, 200)

    #for data in range(len(faces_img_data_color)):
    #    cv2.imshow("face {}".format(data), faces_img_data_color[data])

    #tens_net = tens_train_net.inception_net()
    #output_graph, output_labels = tens_net.load_model("img_data/train", "cnn_tens/validation_save", "cnn_tens/tmp/bottleneck", "final_result", "final_result", 10, 10, 0.01, 50, 100, 10, 100, "cnn_tens/tmp/output_graph.pb", "cnn_tens/tmp/output_labels.txt")
    #tens_net.retrain()

    #net_keras = keras_train_net.seq_net()
    #net_keras.load_model(150, 150, 'img_data/train', 'img_data/validation', weight_location = 'cnn_keras/first_try.h5')
    #model, class_dictionary = net_keras.retrain(50, 16, 2000, 800)
    #print(class_dictionary)
    #print(net_keras.label(model, 'img_data/test/3.jpg'))


    net_keras_v2 = keras_train_net_v2.seq_net()
    net_keras_v2.load_model(150, 150, 'img_data/train', 'img_data/validation', weight_location = 'cnn_keras/first_try.h5')
    model, class_dictionary = net_keras_v2.retrain(50, 16, 42, 42)
    print(class_dictionary)
    print(net_keras_v2.label(model, 'img_data/test/3.jpg'))

if __name__ == "__main__":
    main()
