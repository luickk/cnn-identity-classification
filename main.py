import tensorflow as tf
import numpy as np
import sys
import cv2

from cv import detect_faces
from optparse import OptionParser

# Using pure TF is optional but not recommended
from cnn_tens import train_net as tens_train_net
from cnn_tens import label_pic as tens_label_pic

from cnn_keras import train_net as keras_train_net
from cnn_keras import label_pic as keras_label_pic

def main():


    parser = OptionParser()

    # Only required for labeling - Defines train or label mode
    parser.add_option('-m', '--mode', help='train or label', dest='mode', default = 'label')
    # Required for training or labeling
    parser.add_option('-i', '--input', help='input file', dest='filename', default='example.jpg')
    # Only required for labeling - Enter Model id here
    parser.add_option('-u', '--uid', help='enter model id here')

    (options, args) = parser.parse_args()

    imagePath = options.filename

    img = cv2.imread(imagePath)

    # Defineing haarcascade for face detection
    haar_cascade = 'cv/cascades/haarcascade_frontalface_default.xml'

    model_uuid = ''
    if options.mode == 'train':
        net_keras = keras_train_net.seq_net()
        net_keras.load_model(img_width = 150, img_height = 150, train_data_dir = 'img_data/train', validation_data_dir = 'img_data/valid', model_directory = 'pretrained_models')
        model, class_dictionary, model_uuid = net_keras.retrain(epochs = 50, batch_size = 16, nb_train_samples = 2000, nb_validation_samples = 800)


    if options.mode == 'label':
        model_uuid = options.uid
        faces_img_data, faces_data = detect_faces.detect_faces(haar_cascade , img)
        faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 150, 150)
        for data in range(len(faces_img_data_color)):
            pred, class_dictionary = keras_label_pic.label_pic_no_path(img = faces_img_data_color[data], model_id = model_uuid, model_directory = 'pretrained_models')
            img_name = 'Face {}, Classes: {}'.format(str(pred),class_dictionary)
            cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(img_name, 600,600)
            cv2.imshow(img_name, faces_img_data_color[data])

    print(class_dictionary)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
