"""
Train a CNN on the MNIST handwritten digit dataset.
Run: python train.py
The first run downloads MNIST automatically.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
OUTPUTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pixel values: 0..255 -> 0..1. Add channel dimension for CNN.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)

# CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on the separate test set
test_loss, keras_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
probabilities = model.predict(x_test, batch_size=256, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

metrics = {
    "test_loss": float(test_loss),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    "keras_test_accuracy": float(keras_test_accuracy),
}
print("\n===== TEST METRICS =====")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

report = classification_report(y_test, y_pred, digits=4, zero_division=0)
print("\n===== CLASSIFICATION REPORT =====")
print(report)

with open(OUTPUTS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)
with open(OUTPUTS / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# Save trained model
model.save(MODELS / "mnist_cnn.keras")
print(f"\nSaved model to: {MODELS / 'mnist_cnn.keras'}")

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUTS / "accuracy_plot.png", dpi=150)
plt.close()

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUTS / "loss_plot.png", dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title("MNIST Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)
plt.close()

# Save a few prediction examples
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title(f"Actual: {y_test[i]} | Pred: {y_pred[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUTS / "sample_predictions.png", dpi=150)
plt.close()

print("\nDone. Check the outputs/ folder for graphs and evaluation reports.")
