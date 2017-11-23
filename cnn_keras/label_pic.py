from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from PIL import Image as pil
import numpy as np

def label_pic(img_path, model_path, model_weights_path, img_width, img_height):
    model = load_model(model_path)
    model.load_weights(model_weights_path, by_name=False)

    img_pil = pil.open(img_path)
    img_pil = img_pil.resize((img_width, img_height))
    img_as_array = img_to_array(img_pil)
    class_dictionary = model.class_indices
    return model.predict_classes(img_as_array), class_dictionary

def img_to_array(img):
    x = np.asarray(img, dtype=K.floatx())
    #expand dim for Keras net
    x = np.expand_dims(x / 255, axis=0)
    return x
