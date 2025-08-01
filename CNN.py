# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l_lTeQ2W9OBtJlcypSm5Uu4QINLEyHRr
"""

# 1. Upload your animal image (32x32, e.g. mycat.png)
from google.colab import files
uploaded = files.upload()  # Choose any .png/.jpg you want to test

# 2. Install dependencies (if needed)
!pip install tensorflow pillow matplotlib

# 3. Import libraries and prepare 32x32 animal data
import tensorflow as tf
import numpy as np

animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
class_names = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

animal_train_idx = np.isin(y_train.flatten(), animal_classes)
animal_test_idx = np.isin(y_test.flatten(), animal_classes)
x_train_animal = x_train[animal_train_idx]
y_train_animal = y_train[animal_train_idx]
x_test_animal = x_test[animal_test_idx]
y_test_animal = y_test[animal_test_idx]
y_train_animal = np.array([animal_classes.index(lbl) for lbl in y_train_animal.flatten()])
y_test_animal = np.array([animal_classes.index(lbl) for lbl in y_test_animal.flatten()])

x_train_animal = x_train_animal / 255.0
x_test_animal = x_test_animal / 255.0

# 4. Data augmentation (for more robust learning)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train_animal)

# 5. Early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# 6. Build and train the CNN (sized for 32x32)
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(animal_classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    datagen.flow(x_train_animal, y_train_animal, batch_size=64),
    epochs=50,
    validation_data=(x_test_animal, y_test_animal),
    callbacks=[early_stop]
)

# 7. Plot training/validation accuracy and loss (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()

# 8. Prediction function for 32x32 images
from tensorflow.keras.preprocessing import image

def predict_animal(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    idx = np.argmax(pred)
    predicted_class = class_names[idx]
    confidence = pred[0][idx]
    print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
    # Show the image with prediction as title
    plt.imshow(image.load_img(img_path, target_size=(128,128)))  # Larger for display
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()
    return predicted_class

# 9. Predict on your uploaded image (edit the filename if needed)
predict_animal('cat.jpg')  # Change filename to your uploaded file if needed