from pathlib import Path
import json
import cv2
import os

from joblib import dump
from random import shuffle
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow as tf


def main(repo_path):
    weight_path = repo_path / "model/cats_dogs_classifier.h5"
    model_path = repo_path / "model/cat_dog_simple_model.json"
    data_path = repo_path / "data"
    test_path = data_path / "raw/val"
    test_image = test_path / "cat/cat_ (1).jpg"

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)

    categories = ['cat', 'dog']
    predicted_score = {}
    image_to_predict = cv2.imread(str(test_image))
    image_to_predict = cv2.resize(image_to_predict, (224, 224))
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    class_prob = model.predict(image_to_predict / 255.0)
    percentage = "{:.2}".format(class_prob[0][np.argmax(class_prob[0])])
    predicted_score['confidence_score'] = percentage

    prediction = model.predict(image_to_predict)

    prediction = np.argmax(prediction, axis=1)

    output_word = prediction.tolist()

    check_list = np.asarray(categories)
    predicted_doc = check_list[output_word]
    predicted_doc = predicted_doc.tolist()

    predicted_score['document_type'] = predicted_doc
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(predicted_score))



if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)