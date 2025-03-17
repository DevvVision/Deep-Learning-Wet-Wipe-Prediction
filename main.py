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

IMG_SIZE = 224  # Resize all images for CNN input


# Function to extract bounding boxes from YOLO label files
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


# Generate cropped datasets
os.makedirs("/kaggle/working/cropped", exist_ok=True)

train_df = get_cropped_images_and_labels(train_images_dir, train_labels_dir)
test_df = get_cropped_images_and_labels(test_images_dir, test_labels_dir)
valid_df = get_cropped_images_and_labels(valid_images_dir, valid_labels_dir)

# Convert labels to binary (0 or 1)
train_df["class"] = train_df["class"].apply(lambda x: 1 if x == 1 else 0)
valid_df["class"] = valid_df["class"].apply(lambda x: 1 if x == 1 else 0)
test_df["class"] = test_df["class"].apply(lambda x: 1 if x == 1 else 0)

train_df["class"] = train_df["class"].astype(str)
valid_df["class"] = valid_df["class"].astype(str)
test_df["class"] = test_df["class"].astype(str)

# ImageDataGenerator with Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create Generators
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

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu', input_shape=(224, 224, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

cnn.fit(x=train_generator,epochs=25,validation_data =test_generator )

test_loss, test_accuracy = cnn.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate predictions on the test data
predictions = cnn.predict(test_generator)

# Convert probabilities to binary labels (assuming a 0.5 threshold)
predicted_classes = (predictions > 0.5).astype("int32").reshape(-1)

# Get true labels from the generator (ensure shuffle=False when creating test_generator)
true_classes = test_generator.classes

# Optional: Print detailed classification metrics
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))

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

cnn.save('model1.h5')

# from tensorflow.keras.models import load_model
# model = load_model('my_model.h5')
