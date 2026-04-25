import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tqdm import tqdm
import os
import json
from BE.train_and_eval.training import LiteMHSA

model_dir = "BE/model"
data_dir = "BE/testSet"
plot_dir = "BE/plots"

model_path = os.path.join(model_dir, "best_model.keras")
history_path = os.path.join(model_dir, "historyB2_GenImage_V3.json")
test_dir = os.path.join(data_dir)

os.makedirs(plot_dir, exist_ok=True)

model = tf.keras.models.load_model(
    model_path,
    custom_objects={"LiteMHSA": LiteMHSA}
)

IMG_SIZE = 224
BATCH_SIZE = 64

ds_test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

if os.path.exists(history_path):
    print("Plotting training history...")
    with open(history_path, "r") as f:
        history_data = json.load(f)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_data['loss'], label='Train Loss')
    plt.plot(history_data['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_data['accuracy'], label='Train Accuracy')
    plt.plot(history_data['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "EfficientNetB2_CF_Gen2_Plot.png"))
    plt.close()
else:
    print("Training history file not found. Skipping training plots.")


print("Running predictions on test set...")

y_true = np.concatenate([y.numpy() for _, y in ds_test], axis=0)
y_pred = []
for batch, _ in tqdm(ds_test, desc="Predicting", unit="batch"):
    preds = model.predict(batch, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

y_pred = np.array(y_pred)

test_accuracy = accuracy_score(y_true, y_pred)
print("Test Accuracy:", test_accuracy)


print("Building confusion matrix...")

class_names = ds_test.class_names
plt.title("Confusion Matrix")
plt.savefig(os.path.join(plot_dir, "EfficientNetB2_CF_Gen2_ConfMatrix.png"))
plt.close()

print("All plots saved successfully.")
