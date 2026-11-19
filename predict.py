"""
Predict a handwritten digit using the trained CNN.

Examples:
  python predict.py --index 25
  python predict.py --image my_digit.png

For a custom image: use a clear, single handwritten digit, dark ink on a
light background. The script converts it to MNIST-like grayscale 28x28 input.
"""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mnist_cnn.keras"

def predict_array(model, image_array):
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    probabilities = model.predict(image_array, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    print(f"\nPredicted digit: {digit}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nClass probabilities:")
    for number, probability in enumerate(probabilities):
        print(f"{number}: {probability * 100:.2f}%")

def load_custom_image(path):
    image = Image.open(path).convert("L")
    # MNIST uses light digits on a dark background. Invert if image background
    # is mostly light (typical paper/photo).
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    return np.asarray(image)

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="MNIST test image index (0 to 9999)")
    group.add_argument("--image", type=str, help="Path to a custom digit image")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run 'python train.py' first.")

    model = tf.keras.models.load_model(MODEL_PATH)

    if args.index is not None:
        if not 0 <= args.index < 10000:
            raise SystemExit("Index must be between 0 and 9999.")
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        actual = int(y_test[args.index])
        print(f"Actual digit: {actual}")
        predict_array(model, x_test[args.index])
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")
        predict_array(model, load_custom_image(image_path))

if __name__ == "__main__":
    main()
