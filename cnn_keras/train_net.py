from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import uuid
import numpy as np
import os
import json

from PIL import Image as pil

class seq_net:

    def __init__(self):
        print('Initializing CNN')

    def load_model(self, img_width = 150, img_height = 150, train_data_dir = 'img_data/train', validation_data_dir = 'img_data/valid', model_directory = 'pretrained_models'):
        print('loading_imgs')

        # dimensions of our images.
        self.img_width=  img_width
        self.img_height = img_height
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.model_directory = model_directory

    def retrain(self, epochs = 50, batch_size = 16, nb_train_samples = 2000, nb_validation_samples = 800):

        class_dictionary = None

        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 3)

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
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

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
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode='binary')

        class_dictionary = train_generator.class_indices

        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode='binary')

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        model_uuid = str(uuid.uuid1())

        model_directory_path = self.model_directory + '/' + model_uuid

        if not os.path.exists(model_directory_path):
            os.makedirs(model_directory_path)

        model.save_weights(model_directory_path+'/model_weights.h5')
        model.save(model_directory_path+'/model.h5')

        class_indices_file = open(model_directory_path+'/class_indices_file.txt','w')
        class_indices_file.write(str(class_dictionary))
        class_indices_file.close()

        return model, class_dictionary, model_uuid
