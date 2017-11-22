import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

from PIL import Image as pil

class seq_net:

    def __init__(self):
        print('Initializing CNN')

    def load_model(self, img_width = 150, img_height = 150, train_data_dir = 'img_data/train', validation_data_dir = 'img_data/validation', weight_location = 'cnn_keras/first_try.h5'):
        print('loading_imgs')

        # dimensions of our images.
        self.img_width=  img_width
        self.img_height = img_height
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.weight_location = weight_location



    def retrain(self, epochs = 50, batch_size = 16, nb_train_samples = 20, nb_validation_samples = 20):

        class_dictionary = None


        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 3)

        datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)


        # build the vgg16 model
        model = applications.VGG16(include_top=False, weights='imagenet')

        generator = datagen.flow_from_directory(self.train_data_dir, target_size=(self.img_width, self.img_height),
                                                class_mode='binary',
                                                batch_size=batch_size)  # class_mode=None means our data will only yield
        class_dictionary = generator.class_indices
        # batches of data, no labels, shuffle=False means our data will be in order so first 1000 images will be cats and
        #  next 1000 dogs

        # generates predication for a generator. Steps: total no of batches. Returns a numpy array of predictions
        bottleneck_features_train = model.predict_generator(generator=generator, steps=nb_train_samples // batch_size)
        # saves an array to a binary file
        np.save(file="bottleneck_features_train.npy", arr=bottleneck_features_train)

        generator = datagen.flow_from_directory(self.validation_data_dir, target_size=(self.img_width, self.img_height), batch_size=batch_size,
                                                class_mode='binary')
        bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
        np.save(file="bottleneck_features_validation.npy", arr=bottleneck_features_validation)

        train_data = np.load(file="bottleneck_features_train.npy")
        train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

        validation_data = np.load(file="bottleneck_features_validation.npy")
        validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))  # don't need to tell batch size in input shape
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))
        model.save_weights(self.weight_location)

        return model, class_dictionary


    def label(self, model, img_path):
        img_pil = pil.open(img_path)
        img_pil = img_pil.resize((self.img_width, self.img_height))
        img_as_array = img_to_array(img_pil)
        return model.predict_classes(img_as_array)

def img_to_array(img):
    x = np.asarray(img, dtype=K.floatx())
    #expand dim for Keras net
    x = np.expand_dims(x / 255, axis=0)
    return x
