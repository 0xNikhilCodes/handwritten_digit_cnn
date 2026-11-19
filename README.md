# Handwritten Digit Recognition Using CNN

A backend-only deep-learning project that classifies handwritten digits (0–9)
using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
There is no frontend or web application.

## 1. Requirements

- Python 3.10
- pip
- Internet connection for the first MNIST dataset download

## 2. Setup on Ubuntu / Xubuntu

Open a terminal inside this project folder:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If virtual-environment creation fails, install it with:

```bash
sudo apt update
sudo apt install python3-venv
```

## 3. Train the CNN

```bash
python train.py
```

The script downloads MNIST automatically, preprocesses the images, trains a
CNN, evaluates it on the separate test dataset, and saves the model and results.

## 4. Predict a digit

After training:

```bash
python predict.py --index 25
```

Use any test-set index from 0 to 9999. The output shows the predicted digit,
confidence, and probability for each class.

For a custom image:

```bash
python predict.py --image path/to/digit.png
```

Use one large, centered handwritten digit. The image should contain only one
digit. The simple preprocessing is intended for clear images and may not work
well on photographs with shadows or complex backgrounds.

## 5. Generated files

- `models/mnist_cnn.keras` — trained neural network
- `outputs/metrics.json` — test loss, accuracy, macro precision, recall and F1
- `outputs/classification_report.txt` — per-digit precision, recall and F1
- `outputs/accuracy_plot.png` — training and validation accuracy
- `outputs/loss_plot.png` — training and validation loss
- `outputs/confusion_matrix.png` — test-set confusion matrix
- `outputs/sample_predictions.png` — sample actual and predicted digits

## 6. Neural network architecture

1. Input: 28 × 28 grayscale image
2. Convolution layer: 32 filters, 3 × 3, ReLU
3. Max pooling: 2 × 2
4. Convolution layer: 64 filters, 3 × 3, ReLU
5. Max pooling: 2 × 2
6. Flatten
7. Dense layer: 128 neurons, ReLU
8. Dropout: 30%
9. Output: 10 neurons, Softmax

The model uses Adam optimization and sparse categorical cross-entropy loss.

## 7. Metrics to explain in your report

- **Accuracy:** fraction of all test images classified correctly.
- **Precision:** among images predicted as a particular digit, how many were correct.
- **Recall:** among actual images of a particular digit, how many were found.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** shows which digits are correctly classified and which are confused.
- **Training/validation loss:** helps assess learning and possible overfitting.

The reported precision, recall and F1-score are macro averages across the ten
digit classes. Do not write a result in your report until you have run the
program and recorded the values printed by your own training run.

## 8. Important academic notes

- The training set is split into training and validation portions by Keras.
- The official MNIST test set is kept separate and used only for final evaluation.
- The model is a CNN, satisfying the Unit 3 CNN topic.
- This project performs digit classification; it is not a general handwriting
  recognition system for words or sentences.
