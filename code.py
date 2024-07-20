# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 00:51:06 2024

@author: win 10
"""

import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the CSV file
csv_file_path = 'train.csv'
data = pd.read_csv(csv_file_path)

# Define the path to the image folder
image_folder = 'images'  # Change this to your image folder path

# Initialize lists to hold the images and labels
images = []
labels = []

# Loop over the rows of the CSV file
for index, row in data.iterrows():
    image_path = os.path.join(image_folder, row['image_names'])
    if os.path.exists(image_path):
        # Load the image
        image = Image.open(image_path)
        image = image.resize((128, 128))  # Resize image to 128x128 pixels
        image = img_to_array(image)
        images.append(image)
        labels.append(row['class'])

# Convert lists to numpy arrays
images = np.array(images, dtype='float32') / 255.0
labels = to_categorical(np.array(labels), num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save('gender_classification_model.h5')



# Prediction on test data

import os
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the paths
test_csv_file_path = 'test.csv'
image_folder = 'images'  # Change this to your image folder path
model_path = 'gender_classification_model.h5'  # Path to the saved model

# Load the test CSV file
test_data = pd.read_csv(test_csv_file_path)

# Initialize list to hold the images
test_images = []

# Loop over the rows of the CSV file
for index, row in test_data.iterrows():
    image_path = os.path.join(image_folder, row['image_names'])
    if os.path.exists(image_path):
        # Load the image
        image = Image.open(image_path)
        image = image.resize((128, 128))  # Resize image to 128x128 pixels
        image = img_to_array(image)
        test_images.append(image)

# Convert list to numpy array
test_images = np.array(test_images, dtype='float32') / 255.0

# Load the trained model
model = load_model(model_path)

# Perform predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Add the predictions to the test data
test_data['predicted_class'] = predicted_classes

# Save the results to a new CSV file
output_csv_file_path = 'test_predictions.csv'
test_data.to_csv(output_csv_file_path, index=False)

print(f"Predictions saved to {output_csv_file_path}")
