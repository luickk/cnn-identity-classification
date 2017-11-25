from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from PIL import Image as pil
import numpy as np

def label_pic(img_path, img_width, img_height, model_id, model_directory = 'cnn_keras/models'):
    model = load_model(model_directory+'/'+model_id+'/'+'model.h5')
    model.load_weights(model_directory+'/'+model_id+'/'+'model_weights.h5')

    img_pil = pil.open(img_path)
    img_pil = img_pil.resize((img_width, img_height))
    img_as_array = img_to_array(img_pil)
    class_dictionary_file = open(model_directory+'/'+model_id+'/'+'class_indices_file.txt', 'r')
    class_dictionary = class_dictionary_file.read()
    return model.predict_classes(img_as_array), class_dictionary

def label_pic_no_path(img, model_id, model_directory = 'cnn_keras/models'):
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
