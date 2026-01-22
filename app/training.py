import numpy as np
import tensorflow as tf
import keras
import os
from keras import layers
from keras.applications import EfficientNetV2B1
import json

IMG_SIZE = 224
BATCH_SIZE = 64
DATA_DIR = "CIFAKE"

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

base_model= EfficientNetV2B1(
    include_top=False, 
    weights='imagenet', 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)

h, w, c = base_model.output_shape[1:]
x = layers.Reshape((h * w, c))(x)

x = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=c
)(x, x)

x = layers.LayerNormalization()(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

def unfreeze_model(base_model, num_layers=20):
    for layer in base_model.layers[-num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

unfreeze_model(base_model, num_layers=20)

history = model.fit(ds_train, validation_data=ds_test, epochs=5, class_weight={0:2.0, 1:1.0})

os.makedirs("model", exist_ok=True)
model.save("model/pixelProbeB1_CIFAKE.keras")
with open("model/historyB1_CIFAKE.json", "w") as f:
    json.dump(history.history, f)