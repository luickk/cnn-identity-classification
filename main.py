import tensorflow as tf
import numpy as np
import sys
import cv2
from cv import detect_faces

def main():
    imagePath = sys.argv[1]
    img = cv2.imread(imagePath)
    #img = cv2.VideoCapture(0).read()[1]
    haar_cascade = "cv/cascades/haarcascade_frontalface_default.xml"

    faces_img_data, faces_data = detect_faces.detect_faces(haar_cascade , img)
    faces_img_data_color = detect_faces.img_resize(detect_faces.map_to_color(img, faces_data), 200, 200)

    for data in range(len(faces_img_data_color)):
        cv2.imshow("face {}".format(data), faces_img_data_color[data])


    cv2.waitKey(0)






if __name__ == "__main__":
    main()
