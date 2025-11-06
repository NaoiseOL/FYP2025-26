import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import EfficientNetV2B0
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = 224
BATCH_SIZE = 64
DATA_DIR = r"C:\Users\naois\Downloads\project25\FYP2025-26\images"

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

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

y_true = []
y_pred = []

for images, labels in ds_test.unbatch():
    preds = model.predict(tf.expand_dims(images, axis=0), verbose=0)
    y_true.append(labels.numpy())
    y_pred.append(np.argmax(preds))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ds_test.class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()