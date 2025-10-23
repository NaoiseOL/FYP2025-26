from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load pretrained EfficientNetV2B0 model
model = EfficientNetV2B0(weights='imagenet')  # No need to specify input_shape

# Load and preprocess the image in RGB mode
image_path = 'test2.jpg'
image = load_img(image_path, target_size=(224, 224), color_mode='rgb')  # Ensure RGB
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)

# Make prediction
predictions = model.predict(image_array)

# Decode top 3 predictions
decoded = decode_predictions(predictions, top=3)[0]
print("Top predictions:")
for i, (imagenet_id, label, prob) in enumerate(decoded):
    print(f"{i+1}. {label} ({imagenet_id}): {prob:.4f}")