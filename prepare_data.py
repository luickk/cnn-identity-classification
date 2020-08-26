import cv2
import glob
import os
import numpy as np 

def main():
    data_dir = 'data/data'
    train_data_dir = 'data/data/train'
    validation_data_dir = 'data/data/val'

    preped_train_data_dir = 'data/data_preped/train'
    preped_validation_data_dir = 'data/data_preped/val'

    if not os.path.exists(preped_train_data_dir) and not os.path.exists(preped_validation_data_dir):
        os.makedirs(preped_train_data_dir)
        os.makedirs(preped_validation_data_dir)

    for filename in glob.iglob(train_data_dir+"/**", recursive=True):
        print(filename)
        if os.path.isfile(filename): # filter dirs
            file_ = filename.split('/')[4]
            file_class = filename.split('/')[3]
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            face, found_face = detect_faces("data/haarcascade.xml", img)
            if found_face:
                face = cv2.resize(face, (200, 200))
                if not os.path.exists(preped_train_data_dir+"/"+file_class):
                    os.mkdir(preped_train_data_dir+"/"+file_class)
                print(preped_train_data_dir+"/"+file_class+"/"+file_)
                cv2.imwrite(preped_train_data_dir+"/"+file_class+"/"+file_, face)
            else:
                print('Did not find any faces!')

    
    for filename in glob.iglob(validation_data_dir+"/**", recursive=True):
        print(filename)
        if os.path.isfile(filename): # filter dirs
            file_ = filename.split('/')[4]
            file_class = filename.split('/')[3]
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            face, found_face = detect_faces("data/haarcascade.xml", img)
            if found_face:
                face = cv2.resize(face, (200, 200))
                if not os.path.exists(preped_validation_data_dir+"/"+file_class):
                    os.mkdir(preped_validation_data_dir+"/"+file_class)
                print(preped_validation_data_dir+"/"+file_class+"/"+file_)
                cv2.imwrite(preped_validation_data_dir+"/"+file_class+"/"+file_, face)
            else:
                print('Did not find any faces!')

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
    face = []
    found_face = False
    for (x, y, w, h) in faces:
        cropped = gray[y-200:y+h+400, x-200:x+w+400]
        faces_img_data.append(cropped)
        faces_data.append([x-200, y-200, w+400, h+400])
        # only returning first face found
        face = faces_img_data[0]
        found_face = True
        if len(faces_data) > 1:
            print("found multiple faces!")
    
    return np.array(face), found_face

if __name__ == "__main__":
    main()