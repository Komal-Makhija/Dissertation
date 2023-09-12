#!/usr/bin/env python
# coding: utf-8

# # CNN

# In[1]:


import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[2]:


# Set the path to your dataset directory
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'


# In[3]:


# Define image dimensions and other parameters
image_size = (128, 128)
batch_size = 32


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)


# In[9]:


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

# In[10]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (new_width, new_height))
    flattened_image = resized_image.flatten()
    return flattened_image


# In[11]:


# Load images and extract features
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'
image_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
data = []
labels = []


# In[12]:


for class_idx, class_name in enumerate(image_classes):
    class_directory = os.path.join(dataset_directory, class_name)
    for image_filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, image_filename)
        features = extract_features(image_path, new_width=64, new_height=64)
        data.append(features)
        labels.append(class_idx)


# In[13]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[14]:


# Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)


# In[15]:


# Predict on the test set
y_pred = random_forest_classifier.predict(X_test)


# In[16]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[17]:


# Print classification report
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[18]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# # Knn 

# In[19]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (new_width, new_height))
    flattened_image = resized_image.flatten()
    return flattened_image


# In[20]:


# Load images and extract features
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'
image_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
data = []
labels = []


# In[21]:


for class_idx, class_name in enumerate(image_classes):
    class_directory = os.path.join(dataset_directory, class_name)
    for image_filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, image_filename)
        features = extract_features(image_path, new_width=64, new_height=64)
        data.append(features)
        labels.append(class_idx)


# In[22]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[23]:


# Train a k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)


# In[24]:


# Predict on the test set
y_pred = knn_classifier.predict(X_test)


# In[25]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[26]:


# Print classification report
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[27]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# # SVM

# In[28]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (new_width, new_height))
    flattened_image = resized_image.flatten()
    return flattened_image


# In[29]:


# Load images and extract features
dataset_directory = 'F:/UWL Study Documents/Dissertation/dataset-resized/dataset-resized'
image_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
data = []
labels = []

for class_idx, class_name in enumerate(image_classes):
    class_directory = os.path.join(dataset_directory, class_name)
    for image_filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, image_filename)
        features = extract_features(image_path, new_width=64, new_height=64)
        data.append(features)
        labels.append(class_idx)


# In[30]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[31]:


# Train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
svm_classifier.fit(X_train, y_train)


# In[32]:


# Predict on the test set
y_pred = svm_classifier.predict(X_test)


# In[33]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[34]:


# Print classification report
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[35]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[36]:


# Resize images to (128, 128)
X_train_cnn = [cv2.resize(img, (128, 128)) for img in X_train]
X_test_cnn = [cv2.resize(img, (128, 128)) for img in X_test]


# In[37]:


# Convert lists to arrays
X_train_cnn = np.array(X_train_cnn)
X_test_cnn = np.array(X_test_cnn)


# In[38]:


# Initialize classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
svm_classifier = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)


# In[39]:


# Initialize Keras-based CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[40]:


# Define a dictionary to store the classifiers
classifiers = {
    'Random Forest': rf_classifier,
    'K-Nearest Neighbors': knn_classifier,
    'SVM': svm_classifier,
    'CNN': model
}


# In[41]:


# Initialize dictionaries to store accuracy and confusion matrices
accuracy_scores = {}
confusion_matrices = {}


# In[42]:


# Loop through classifiers and evaluate each one
for classifier_name, classifier in classifiers.items():
    if classifier_name == 'CNN':
        # Reshape the data for the CNN
        X_train_cnn = np.array(X_train).reshape(-1, 64, 64, 3)
        X_test_cnn = np.array(X_test).reshape(-1, 64, 64, 3)
        # One-hot encode labels for the CNN
        y_train_cnn = np.eye(6)[y_train]
        y_test_cnn = np.eye(6)[y_test]
    else:
        classifier.fit(X_train, y_train)  # Train the classifier
        y_pred = classifier.predict(X_test)  # Make predictions on the test data
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        accuracy_scores[classifier_name] = accuracy  # Store accuracy in the dictionary


# In[43]:


# Train the CNN
classifier.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_split=0.2)


# In[44]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_scores[classifier_name] = accuracy


# In[45]:


# Calculate and store confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_matrices[classifier_name] = conf_matrix


# In[46]:


# Compare accuracy scores
for classifier_name, accuracy in accuracy_scores.items():
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")


# In[47]:


# Plot confusion matrices
plt.figure(figsize=(12, 10))
for i, (classifier_name, conf_matrix) in enumerate(confusion_matrices.items(), 1):
    plt.subplot(2, 2, i)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {classifier_name}")
plt.tight_layout()
plt.show()

