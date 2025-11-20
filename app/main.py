import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your custom model
model = tf.keras.models.load_model("model/pixelProbeB0.keras")


class_labels = ["real", "fake"]

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

    preds = model.predict(x)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds)
    return predicted_class, confidence

image_path = 'images/train/real/2irnom9vxi5a1.png'
pred_class, conf = classify_image(image_path)
print(f"Predicted: {pred_class} (confidence: {conf:.2f})")