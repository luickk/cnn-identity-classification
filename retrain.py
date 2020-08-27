from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
import keras
import uuid
import numpy as np
import os
import json

from PIL import Image as pil


def main():
    model, class_dictionary, model_uuid = reTrain(model_id = "7c88aeaa-e84a-11ea-a288-faffc2003e2a", img_width = 200, img_height = 200, retrain_data_dir = 'data/data/retrain', model_directory_path = 'data/trainedModels',
                                                epochs = 200, batch_size = 3, nb_train_samples = 19, nb_validation_samples = 5)

def reTrain(model_id, img_width, img_height, retrain_data_dir, model_directory_path,
                epochs, batch_size, nb_train_samples, nb_validation_samples):

    class_dictionary = None

    input_shape = (img_width, img_height, 3)

    # this is the augmentation configuration we will use for training
    retrain_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    retrain_generator = retrain_datagen.flow_from_directory(
        retrain_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    class_dictionary = retrain_generator.class_indices
    
    model = load_model(model_directory_path+'/model_'+model_id+'.h5')


    model.fit_generator(
        retrain_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_steps=nb_validation_samples // batch_size)

    if not os.path.exists(model_directory_path):
        os.makedirs(model_directory_path)

    model_uuid = str(uuid.uuid1())
    print("Model id: " + model_uuid)
    model.save(model_directory_path+'/model_'+model_uuid+'.h5')

    class_indices_file = open(model_directory_path+'/class_indices_file.txt','w')
    class_indices_file.write(str(class_dictionary))
    class_indices_file.close()

    return model, class_dictionary, model_uuid


if __name__ == "__main__":
    main()