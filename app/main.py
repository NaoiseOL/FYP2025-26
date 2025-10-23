import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Classify the image
    preds = model.predict(x)
    return preds

# Classify a new image
image_path = '001.jpg'
preds = classify_image(image_path)
print('Predicted:', tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0])