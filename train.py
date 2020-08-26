from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import uuid
import numpy as np
import os
import json

from PIL import Image as pil


def main():
    model, class_dictionary, model_uuid = initialTrain(img_width = 200, img_height = 200, train_data_dir = 'data/data/train', validation_data_dir = 'data/data/val', model_directory_path = 'data/trainedModels',
                                                epochs = 500, batch_size = 3, nb_train_samples = 19, nb_validation_samples = 5)

def initialTrain(img_width, img_height, train_data_dir, validation_data_dir, model_directory_path,
                epochs, batch_size, nb_train_samples, nb_validation_samples):

    class_dictionary = None

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    class_dictionary = train_generator.class_indices

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_dictionary.keys())))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
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