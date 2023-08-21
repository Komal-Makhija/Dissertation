#!/usr/bin/env python
# coding: utf-8

# # CNN

# In[1]:


import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(gray_image, (new_width, new_height))
    
    # Flatten the image pixels as features
    features = resized_image.flatten()
    return features


# In[3]:


# Set the path to your dataset directory
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'


# In[4]:


# Define image dimensions and other parameters
image_size = (128, 128)
batch_size = 32


# In[5]:


# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255.0,    # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Split data into training and validation sets
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)


# In[6]:


# Load and preprocess images using the generator
train_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# In[7]:


# Create a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 output classes
])


# In[8]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[9]:


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)


# In[10]:


# Evaluate the model
test_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred_classes == y_true)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Plot confusion matrix as a heatmap
class_names = list(test_generator.class_indices.keys())
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# # Random Forest

# In[11]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (new_width, new_height))
    flattened_image = resized_image.flatten()
    return flattened_image


# In[12]:


# Load images and extract features
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'
image_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
data = []
labels = []


# In[13]:


for class_idx, class_name in enumerate(image_classes):
    class_directory = os.path.join(dataset_directory, class_name)
    for image_filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, image_filename)
        features = extract_features(image_path, new_width=64, new_height=64)
        data.append(features)
        labels.append(class_idx)


# In[14]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[15]:


# Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)


# In[16]:


# Predict on the test set
y_pred = random_forest_classifier.predict(X_test)


# In[17]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[18]:


# Print classification report
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[19]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# # Knn 

# In[20]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (new_width, new_height))
    flattened_image = resized_image.flatten()
    return flattened_image


# In[21]:


# Load images and extract features
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'
image_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
data = []
labels = []


# In[22]:


for class_idx, class_name in enumerate(image_classes):
    class_directory = os.path.join(dataset_directory, class_name)
    for image_filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, image_filename)
        features = extract_features(image_path, new_width=64, new_height=64)
        data.append(features)
        labels.append(class_idx)


# In[23]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[24]:


# Train a k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)


# In[25]:


# Predict on the test set
y_pred = knn_classifier.predict(X_test)


# In[26]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[27]:


# Print classification report
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[28]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

