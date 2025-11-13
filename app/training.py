import numpy as np
import tensorflow as tf
import keras
import os
from keras import layers
from keras.applications import EfficientNetV2B0

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
)

ds_test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    label_mode="int"
)

class_names = ds_train.class_names
NUM_CLASSES = len(class_names)

base_model= EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    ds_train,
    validation_data = ds_test,
    epochs=10
)

model.fit(ds_train, validation_data=ds_test, epochs=10)

os.makedirs("model/pixeProbB0.keras", exist_ok=True)
model.save("model/pixeProbB0.keras")