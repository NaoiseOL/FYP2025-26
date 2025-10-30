import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

model = EfficientNetV2B0(weights='imagenet')

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

image_path='images/train/real/2irnom9vxi5a1.png'
preds = classify_image(image_path)
print("Predicted: ", tf.keras.applications.efficientnet_v2.decode_predictions(preds, top=3)[0])