from pathlib import Path
import cv2
import os

from joblib import dump
from random import shuffle
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow as tf

batch_size = 1
epochs = 3
data_augmentation = True
num_classes = 2
img_width, img_height = 224,224

selected_names = ["cat", "dog"]
assoc_table = dict([(k, np.identity(len(selected_names))[i]) for (i, k) in enumerate(selected_names)])

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

def main(repo_path):
    data_path = repo_path / "data"
    train_path = data_path / "raw/train"
    test_path = data_path / "raw/val"
    model_path = repo_path / 'model'
    weight_path = model_path / "cat_dog_simple_model.json"
    train_generator = train_datagen.flow_from_directory(train_path, target_size=(img_width, img_height), batch_size=1,
                                                        shuffle=True)
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(img_width, img_height), batch_size=1,
                                                      shuffle=True)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    def save_model(model):
        model_json = model.to_json()

        with open(weight_path, "w") as json_file:
            json_file.write(model_json)

    save_model(model)

    model.compile(
            optimizer='adam',
            loss="categorical_crossentropy",
            metrics=['accuracy'])

    train_samples = len(train_generator)
    test_samples = len(test_generator)

    model.fit_generator(train_generator, steps_per_epoch=train_samples // batch_size, epochs=epochs,
                                         validation_data=test_generator, validation_steps=test_samples // batch_size)

    #dump(trained_model, repo_path / "model/model.joblib")


    model.save_weights(repo_path / "model/cats_dogs_classifier.h5")



if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)