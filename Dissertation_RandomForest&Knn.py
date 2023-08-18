#!/usr/bin/env python
# coding: utf-8

# # Random Forest Algorithm

# In[5]:


get_ipython().system('pip install opencv-python')


# In[6]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(gray_image, (new_width, new_height))
    
    hog_features = cv2.HOGDescriptor().compute(resized_image)
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return np.concatenate((hog_features, color_hist.flatten()))


# In[8]:


# Load images and extract features
train_directory = 'F:/UWL Study Documents/Dissertation/archive/DATASET/DATASET/TRAIN'
test_directory = 'F:/UWL Study Documents/Dissertation/archive/DATASET/DATASET/TEST'
image_classes = ['O', 'R']
data = []
labels = []

data = []
labels = []

for class_name in image_classes:
    train_class_directory = os.path.join(train_directory, class_name)
    test_class_directory = os.path.join(test_directory, class_name)
    
    for image_filename in os.listdir(train_class_directory):
        image_path = os.path.join(train_class_directory, image_filename)  # Corrected image path
        features = extract_features(image_path, new_width=64, new_height=128)
        data.append(features)
        labels.append(class_name)  # Use class_name as label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[9]:


# Preprocessing
new_width, new_height = 64, 128


# In[10]:


# Resize and extract features for each image
processed_data = []
for features, label in zip(X_train, y_train):
    processed_data.append((features, label))


# In[11]:


# Split the processed data into features and labels
X_train_processed = [data[0] for data in processed_data]
y_train_processed = [data[1] for data in processed_data]


# In[12]:


# Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_processed, y_train_processed)


# In[13]:


# Preprocess test data
processed_test_data = []
for features, label in zip(X_test, y_test):
    processed_test_data.append((features, label))


# In[14]:


# Split the processed test data into features and labels
X_test_processed = [data[0] for data in processed_test_data]
y_test_processed = [data[1] for data in processed_test_data]


# In[15]:


# Predict on the test set
y_pred = random_forest_classifier.predict(X_test_processed)


# In[16]:


# Calculate accuracy
accuracy = accuracy_score(y_test_processed, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[17]:


# Print classification report
class_names = ['Organic', 'Recyclable']
print(classification_report(y_test_processed, y_pred, target_names=class_names))


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test_processed, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# # KNN Algorithm

# In[18]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[19]:


# Preprocessing steps
# 1. Handling missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[20]:


# 2. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[21]:


# 3. Dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[22]:


# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors


# In[23]:


# Train the classifier
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
class_names = ['Organic', 'Recyclable']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[27]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[28]:


# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

