import os
import json
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Resizing
from keras.applications import EfficientNetV2B1
from keras.applications.efficientnet_v2 import preprocess_input


class LiteMHSA(layers.Layer):
    def __init__(self, dim, heads=4, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.reduction = reduction
        self.inner = dim // reduction

        self.qkv = layers.Conv2D(self.inner * 3, 1, padding="same")
        self.proj = layers.Conv2D(dim, 1, padding="same")

    def get_config(self): 
        config = super().get_config() 
        config.update({ 
            "dim": self.dim, 
            "heads": self.heads, 
            "reduction": self.reduction 
            })
        return config

    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        def reshape_heads(t):
            t = tf.reshape(t, [B, H * W, self.heads, self.inner // self.heads])
            return tf.transpose(t, [0, 2, 1, 3])

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn / tf.math.sqrt(tf.cast(self.inner // self.heads, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, H, W, self.inner])

        return self.proj(out) 

IMG_SIZE = 224
BATCH_SIZE = 64
DATA_DIR = "CIFAKE"

train_dir = f"{DATA_DIR}/train"
test_dir = f"{DATA_DIR}/test"

ds_train = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

ds_test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = ds_train.class_names
NUM_CLASSES = len(class_names)

img_augmentation_layers = keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip(),
    layers.RandomContrast(0.1),
], name="img_augmentation")


base_model = EfficientNetV2B1(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Resizing(IMG_SIZE, IMG_SIZE)(inputs)
x = img_augmentation_layers(x)              #Aims to increase accuracy on images outside dataset
x = preprocess_input(x)

x = base_model(x, training=False)

h, w, c = base_model.output_shape[1:]

x = LiteMHSA(dim=c, heads=4, reduction=4)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)


def unfreeze_model(base_model, num_layers=20):
    for layer in base_model.layers[-num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

unfreeze_model(base_model, num_layers=20)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


if __name__ == "__main__":      #Ensure this is enclosed in wrapper or else code will run with import in analysis
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=5
    )

    os.makedirs("model", exist_ok=True)
    model.save("model/pixelProbeB1_CIFAKE_V2.keras")

    with open("model/historyB1_CIFAKE_V2.json", "w") as f:
        json.dump(history.history, f)
