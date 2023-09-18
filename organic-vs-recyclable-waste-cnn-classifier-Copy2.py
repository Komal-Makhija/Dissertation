#!/usr/bin/env python
# coding: utf-8

# # Organic Vs Recyclable Waste Classification

# In this notebook we will deal with a dataset containing 25,000 images of waste. Our task is to build a model to classify this waste into organic waste and recyclable waste. We will experiment with CNN classifiers in order to achieve this task

# In[87]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from custom_cnn import CustomCNN  # Import custom CNN model

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ## Parse Data directories

# In[3]:


def get_img_paths(path):
    paths = []
    labels = []
    for label in os.listdir(path):
        img_dir = os.path.join(path, label)
        for img in os.listdir(img_dir):
            paths.append(os.path.join(img_dir, img))
            labels.append(label)

    return pd.DataFrame({'path':paths, 'label':labels})


# ## Training Paths

# In[6]:


train = get_img_paths("F:/UWL Study Documents/Dissertation/archive/DATASET/DATASET/TRAIN")
train.head()


# In[7]:


train.info()


# ## Test Paths

# In[10]:


test = get_img_paths("F:/UWL Study Documents/Dissertation/archive/DATASET/DATASET/TEST")
test.head()


# In[11]:


test.info()


# ## Label Encoding

# In[12]:


conversion = {'O': 0, 'R': 1}

train.label = train.label.map(conversion)
test.label = test.label.map(conversion)

train.head()


# ## Dataset Generator

# In[13]:


class WasteData(Dataset):
    def __init__(self, dir_lbl, transform=None):
        self.dir_lbl = dir_lbl
        self.transform = transform

    def __len__(self):
        return len(self.dir_lbl)

    def __getitem__(self, idx):
        img_dir_lbl = self.dir_lbl.iloc[idx]
        img_dir = img_dir_lbl.path
        label = img_dir_lbl.label
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label


# In[14]:


data_transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225])
])


# In[15]:


train_dataset = WasteData(train, data_transform)
train_dataset

train_size = int(0.9 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])


# In[16]:


test_dataset = WasteData(test, data_transform)
test_dataset


# In[17]:


batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# ## Display Preprocessed Image Sample

# In[18]:


for i in range(5):
    train_features, train_labels = next(iter(train_dataloader))
    img = train_features[0].squeeze().numpy().transpose((1, 2, 0))

    label = train_labels[0]
    print(f"Label {i+1}: {label}")
    plt.imshow(img)
    plt.show()


# ## Enable GPU

# In[19]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


# ## Construct Model

# In[20]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
    
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
            
        self.fc1 = nn.Linear(64*24*24, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.fc4 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    
net = Net().to(device)


# In[21]:


import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)


# ## Training

# We will use early stopping to get the best validation errors. this happens in 4 epochs

# In[22]:


train_loss = []
val_loss = []
epochs = 4
for epoch in range(epochs): 
    epoch_loss = 0.0
    epoch_loss_val = 0.0
    running_loss = 0.0
    print('Training:')
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].float().to(device)

        optimizer.zero_grad()

        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            loss = running_loss / 10
            epoch_loss += loss
            print(f'\t[{epoch + 1}, {i + 1:5d}] loss: {loss:.6f}')
            running_loss = 0.0
     
    print('Validation:')
    running_loss_valid = 0.0
    for i, data in enumerate(valid_dataloader, 0):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].float().to(device)

            outputs = net(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss_valid += loss.item()
            if i % 10 == 9:
                loss = running_loss_valid / 10
                epoch_loss_val += loss
                print(f'\t[{epoch + 1}, {i + 1:5d}] loss: {loss:.6f}')
                running_loss_valid = 0.0
                
    train_loss.append(epoch_loss)
    val_loss.append(epoch_loss_val)

print('Finished Training and Validation')


# In[25]:


save_path = 'cnn_model.pth'


# In[26]:


# Save the trained model
torch.save(net.state_dict(), save_path)


# In[23]:


plt.figure(figsize=(20,6));
sns.lineplot(x=list(range(epochs)), y=train_loss)
sns.lineplot(x=list(range(epochs)), y=val_loss)
plt.legend(['Training loss', 'Validation loss']) 


# ## Testing

# In[24]:


correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data[0].to(device), data[1].float().to(device)
        outputs = net(images)
        predicted = torch.round(outputs.data).squeeze()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


# # Random Forest

# In[27]:


get_ipython().system('pip install opencv-python')


# In[28]:


# Function to extract features from an image
def extract_features(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(gray_image, (new_width, new_height))
    
    hog_features = cv2.HOGDescriptor().compute(resized_image)
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return np.concatenate((hog_features, color_hist.flatten()))


# In[31]:


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


# In[32]:


# Preprocessing
new_width, new_height = 64, 128


# In[33]:


# Resize and extract features for each image
processed_data = []
for features, label in zip(X_train, y_train):
    processed_data.append((features, label))


# In[34]:


# Split the processed data into features and labels
X_train_processed = [data[0] for data in processed_data]
y_train_processed = [data[1] for data in processed_data]


# In[35]:


# Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_processed, y_train_processed)


# In[36]:


# Preprocess test data
processed_test_data = []
for features, label in zip(X_test, y_test):
    processed_test_data.append((features, label))


# In[37]:


# Split the processed test data into features and labels
X_test_processed = [data[0] for data in processed_test_data]
y_test_processed = [data[1] for data in processed_test_data]


# In[38]:


# Predict on the test set
y_pred = random_forest_classifier.predict(X_test_processed)


# In[39]:


# Calculate accuracy
accuracy = accuracy_score(y_test_processed, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[40]:


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


# # Knn Algorithm

# In[41]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[42]:


# Preprocessing steps
# 1. Handling missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[43]:


# 2. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[44]:


# 3. Dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[45]:


# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  


# In[46]:


# Train the classifier
knn_classifier.fit(X_train, y_train)


# In[47]:


# Predict on the test set
y_pred = knn_classifier.predict(X_test)


# In[48]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[49]:


# Print classification report
class_names = ['Organic', 'Recyclable']
print(classification_report(y_test, y_pred, target_names=class_names))


# In[50]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[51]:


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


# # Prediction

# In[72]:


# Define the class labels
class_labels = ['Organic', 'Recyclable']


# In[73]:


# Load the saved model
saved_model_path = 'cnn_model.pth'
model = Net()  # Replace 'Net' with the name of CNN class
model.load_state_dict(torch.load(saved_model_path))
model.eval()


# In[74]:


# Define the transformation for test images
data_transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[75]:


# Load and preprocess the test image-Input img of carrot
image_path = 'F:/UWL Study Documents/Dissertation/ds1_test_img.jpg'  
image = Image.open(image_path).convert('RGB')
image = data_transform(image).unsqueeze(0)


# In[76]:


# Make a prediction
with torch.no_grad():
    output = model(image)
    predicted_class = output.round().item()  # Get the predicted class (0 or 1)


# In[77]:


# Get the corresponding class label
predicted_label = class_labels[int(predicted_class)]

print(f'Predicted class: {predicted_label}')


# Got the correct prediction for organic class so now trying for recyclable class

# In[81]:


# Load and preprocess the test image-Input img of straws
image_path = 'F:/UWL Study Documents/Dissertation/ds1_test_img2.jpg'  
image = Image.open(image_path).convert('RGB')
image = data_transform(image).unsqueeze(0)


# In[82]:


# Make a prediction
with torch.no_grad():
    output = model(image)
    predicted_class = output.round().item()  # Get the predicted class (0 or 1)


# In[83]:


# Get the corresponding class label
predicted_label = class_labels[int(predicted_class)]

print(f'Predicted class: {predicted_label}')


# In[84]:


# Load and preprocess the test image-Input img of can
image_path = 'F:/UWL Study Documents/Dissertation/ds1_test_img3.jpg'  
image = Image.open(image_path).convert('RGB')
image = data_transform(image).unsqueeze(0)


# In[85]:


# Make a prediction
with torch.no_grad():
    output = model(image)
    predicted_class = output.round().item()  # Get the predicted class (0 or 1)


# In[86]:


# Get the corresponding class label
predicted_label = class_labels[int(predicted_class)]

print(f'Predicted class: {predicted_label}')


# # Comparision between the different models and algorithms used

# In[88]:


# Define a function to evaluate a model and return performance metrics
def evaluate_model(model, dataloader):
    y_true = []
    y_pred = []

    for data in dataloader:
        images, labels = data[0].to(device), data[1].float().to(device)
        outputs = model(images)
        predicted = torch.round(outputs.data).squeeze()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


# In[89]:


# Evaluate the CNN model
cnn_accuracy, cnn_precision, cnn_recall, cnn_f1 = evaluate_model(net, test_dataloader)


# In[90]:


# Evaluate the K-Nearest Neighbors model
knn_accuracy = accuracy_score(y_true_knn, y_pred_knn)


# In[ ]:




