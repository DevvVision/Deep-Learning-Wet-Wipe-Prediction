# Deep-Learning-Wet-Wipe-Prediction
YOLO-Based Image Cropping & CNN Binary Classification
This repository contains code that processes images annotated in YOLO format by cropping objects from the images, creates training, validation, and test datasets using TensorFlow’s ImageDataGenerator, and trains a Convolutional Neural Network (CNN) to perform binary classification.

Project Overview
Dataset Structure:
The dataset is divided into three main folders: train, test, and valid.
Each folder contains:

An images directory with JPG images.
A labels directory with corresponding YOLO-formatted text files.
YOLO Annotation Format:
Each label file contains lines with the format:

arduino
Copy
Edit
class_id x_center y_center width height
These values are normalized to the image dimensions. The code converts these normalized values into pixel coordinates to crop objects.

Binary Classification:
The project assumes a binary classification task (two classes, 0 and 1).
Labels are converted to binary strings so that ImageDataGenerator can correctly load the data using class_mode="binary".

Code Description
1. Importing Libraries and Defining Paths
python
Copy
Edit
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'train'
test_dir = 'test'
valid_dir = 'valid'

train_images_dir = os.path.join(train_dir, "images")
test_images_dir = os.path.join(test_dir, "images")
valid_images_dir = os.path.join(valid_dir, "images")

train_labels_dir = os.path.join(train_dir, "labels")
test_labels_dir = os.path.join(test_dir, "labels")
valid_labels_dir = os.path.join(valid_dir, "labels")

IMG_SIZE = 224  # All images are resized to 224x224 for CNN input
Purpose:
Sets up necessary libraries and defines paths to the images and label files in the dataset folders.
2. Cropping Images Based on YOLO Annotations
python
Copy
Edit
def get_cropped_images_and_labels(image_dir, label_dir):
    data = []

    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        with open(label_path, "r") as file:
            for line in file.readlines():
                class_id, x_center, y_center, w, h = map(float, line.strip().split())

                # Convert YOLO format to pixel coordinates
                x_center *= width
                y_center *= height
                w *= width
                h *= height

                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Crop the image
                cropped_object = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

                if cropped_object.size == 0:
                    continue  # Skip invalid crops

                # Save cropped image temporarily for ImageDataGenerator
                cropped_path = f"/kaggle/working/cropped/{filename}"
                cv2.imwrite(cropped_path, cv2.resize(cropped_object, (IMG_SIZE, IMG_SIZE)))

                # Append binary labels (assuming two classes: 0 and 1)
                data.append([cropped_path, int(class_id)])

    return pd.DataFrame(data, columns=["filepath", "class"])
Purpose:
Iterates over images, reads corresponding YOLO labels.
Converts normalized bounding box coordinates into pixel values.
Crops the objects from the images.
Saves the resized cropped images to a temporary directory (/kaggle/working/cropped).
Returns a Pandas DataFrame with file paths and labels.
3. Preparing the Datasets
python
Copy
Edit
os.makedirs("/kaggle/working/cropped", exist_ok=True)

train_df = get_cropped_images_and_labels(train_images_dir, train_labels_dir)
test_df = get_cropped_images_and_labels(test_images_dir, test_labels_dir)
valid_df = get_cropped_images_and_labels(valid_images_dir, valid_labels_dir)

# Convert labels to binary (0 or 1) and then to strings
train_df["class"] = train_df["class"].apply(lambda x: 1 if x == 1 else 0).astype(str)
valid_df["class"] = valid_df["class"].apply(lambda x: 1 if x == 1 else 0).astype(str)
test_df["class"] = test_df["class"].apply(lambda x: 1 if x == 1 else 0).astype(str)
Purpose:
Creates DataFrames for training, testing, and validation datasets. Converts numeric class labels into strings so that the data generator can use class_mode="binary".
4. Data Augmentation with ImageDataGenerator
python
Copy
Edit
# Create ImageDataGenerator instances with data augmentation for training
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generate data generators from the DataFrames
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filepath",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="binary"
)

valid_generator = test_datagen.flow_from_dataframe(
    valid_df,
    x_col="filepath",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filepath",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Print dataset info
print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}, Test samples: {len(test_df)}")
Purpose:
Applies data augmentation (rotation, zoom, horizontal flip) to the training set.
Rescales pixel values to the range [0, 1] for both training and testing.
Creates generators that read data directly from the DataFrame for efficient batch processing.
5. Building and Training the CNN Model
python
Copy
Edit
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu', input_shape=(224, 224, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=train_generator, epochs=25, validation_data=test_generator)
Purpose:
Defines a CNN model with two convolutional layers followed by max pooling.
Uses a Dense layer with dropout to reduce overfitting.
The final Dense layer with a sigmoid activation outputs a probability for binary classification.
The model is compiled with the Adam optimizer and binary crossentropy loss.
Trains the model using the training data generator and validates on the test generator.
6. Evaluating the Model
python
Copy
Edit
test_loss, test_accuracy = cnn.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
Purpose:
Evaluates the trained model on the test dataset and prints loss and accuracy metrics.
7. Generating Predictions and Reporting Metrics
python
Copy
Edit
# Generate predictions on the test data
predictions = cnn.predict(test_generator)

# Convert probabilities to binary labels using a threshold of 0.5
predicted_classes = (predictions > 0.5).astype("int32").reshape(-1)

# Get true labels from the generator
true_classes = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))
Purpose:
Generates predictions on the test data.
Converts prediction probabilities to binary class labels.
Calculates and prints the confusion matrix and classification report (precision, recall, and f1-score).
8. Visualizing Predictions
python
Copy
Edit
import matplotlib.pyplot as plt

images, labels = next(test_generator)
preds = cnn.predict(images)
pred_classes = (preds > 0.5).astype("int32")

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i])
    plt.title(f"Pred: {pred_classes[i][0]}")
    plt.axis("off")
plt.show()
Purpose:
Displays a batch of test images with their predicted labels for visual inspection.
9. Saving the Model
python
Copy
Edit
cnn.save('model1.h5')
Purpose:
Saves the trained model in HDF5 format for later use. You can load this model using TensorFlow’s load_model function.
How to Run
Set Up Directory Structure:
Ensure your repository has the following structure:

bash
Copy
Edit
├── train
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels
Install Requirements:
Make sure you have Python 3.x installed along with the necessary libraries:

bash
Copy
Edit
pip install numpy opencv-python matplotlib pandas tensorflow scikit-learn
Run the Notebook/Script:
Execute the code in your preferred environment (Jupyter Notebook, Google Colab, or as a standalone Python script).

Explore Results:
The code prints dataset statistics, training progress, evaluation metrics, and visualizes sample predictions.
