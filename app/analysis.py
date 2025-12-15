import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import json

# --- Paths ---
model_dir = "model"
data_dir = "images"
plot_dir = "plots"

model_path = os.path.join(model_dir, "pixelProbeB1_V3.keras")
history_path = os.path.join(model_dir, "historyB1_V3.json")
test_dir = os.path.join(data_dir, "test")

# Ensure plots directory exists
os.makedirs(plot_dir, exist_ok=True)

# --- Load model ---
model = tf.keras.models.load_model(model_path)

IMG_SIZE = 224
BATCH_SIZE = 64

# --- Load test dataset ---
ds_test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# Training Plots
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history_data = json.load(f)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history_data['loss'], label='Train Loss')
    plt.plot(history_data['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history_data['accuracy'], label='Train Accuracy')
    plt.plot(history_data['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "EfficientNetB1_V3_Plot.png"))
    plt.show()
else:
    print("Training history file not found. Skipping training plots.")

#Confusion Matrix
y_true = []
y_pred = []

for images, labels in ds_test.unbatch():
    preds = model.predict(tf.expand_dims(images, axis=0), verbose=0)
    y_true.append(labels.numpy())
    y_pred.append(np.argmax(preds))

class_names = ds_test.class_names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(plot_dir, "EfficientNetB1_V3_ConfMatrix.png"))
plt.show()
