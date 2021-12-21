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

    model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

    x = model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=preds)


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


    model.save_weights(repo_path / "model/cats_dogs_classifier_vgg16.h5")



if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)