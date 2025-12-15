import numpy as np
import tensorflow as tf
import keras
import os
from keras import layers
from keras.applications import EfficientNetV2B1
import json

IMG_SIZE = 224
BATCH_SIZE = 64
DATA_DIR = "images"

train_dir=f"{DATA_DIR}/train"
test_dir=f"{DATA_DIR}/test"

ds_train = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    label_mode="int"
).map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

ds_test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    label_mode="int"
).map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

print("Class names:,", ds_test.class_names)