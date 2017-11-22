import tensorflow as tf
import numpy as np
import sys
import cv2
from cnn import train_net
from cnn import label_pic
from cv import detect_faces

def main():
    imagePath = sys.argv[1]
    img = cv2.imread(imagePath)
    #img = cv2.VideoCapture(0).read()[1]
    haar_cascade = "cv/cascades/haarcascade_frontalface_default.xml"

    faces_img_data, faces_data = detect_faces.detect_faces(haar_cascade , img)
    faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 200, 200)

    #for data in range(len(faces_img_data_color)):
    #    cv2.imshow("face {}".format(data), faces_img_data_color[data])

    net = train_net.inception_net()

    output_graph, output_labels = net.load_imgs("img_data", "validation_save", "tmp/bottleneck", "final_result", "final_result", 10, 10, 0.01, 50, 100, 10, 100, "tmp/output_graph.pb", "tmp/output_labels.txt")

    net.retrain()

    print('---------', output_graph, output_labels)

    print(label_pic.label_pic(imagePath, output_graph, output_labels))



if __name__ == "__main__":
    main()
