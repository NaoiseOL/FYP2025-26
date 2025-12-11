from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model/pixelProbeB1_V2.keras")
class_labels = ["real", "fake"]

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    predicted_class = class_labels[np.argmax(preds[0])]
    confidence = float(preds[0][np.argmax(preds[0])])
    return predicted_class, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    pred_class, conf = classify_image(filepath)
    return jsonify({"prediction": pred_class, "confidence": conf})

if __name__ == "__main__":
    app.run(debug=True)
