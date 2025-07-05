# Task 2 - Image Classification using TensorFlow

## Project Overview

This project implements a Deep Learning model using **TensorFlow** for **image classification** on the **MNIST dataset** (handwritten digits). The model achieves high accuracy and is trained for 5 epochs.

---

## 🔧 Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Matplotlib

---

## 📊 Model Performance

- **Training Accuracy (Final):** ~97.6%
- **Validation Accuracy (Final):** ~97.7%
- **Test Accuracy:** `0.9764999747276306`

![Training vs Validation Accuracy](accuracy_plot.png)

---

## 🧠 Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
