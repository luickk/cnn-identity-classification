import cv2
import numpy as np


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
