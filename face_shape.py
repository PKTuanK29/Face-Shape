# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Đảm bảo thư viện matplotlib đã được cài đặt
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns
from termcolor import colored
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from keras.utils import plot_model
from tensorflow.keras import optimizers
from pathlib import Path
import os
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the dataset directories
DATA_DIR = '/content/drive/MyDrive/archive/FaceShape Dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'training_set')
TEST_DIR = os.path.join(DATA_DIR, 'testing_set')

# Function to check the number of classes
def num_of_classes(folder_dir, folder_name):
    classes = [class_name for class_name in os.listdir(folder_dir)]
    print(f'Number of classes in {folder_name} folder: {len(classes)}')

# Check number of classes in train and test directories
num_of_classes(TRAIN_DIR, 'train')
num_of_classes(TEST_DIR, 'test')

# Count the number of images per class in training set
classes = [class_name for class_name in os.listdir(TRAIN_DIR)]
count = []
for class_name in classes:
    count.append(len(os.listdir(os.path.join(TRAIN_DIR, class_name))))

# Plot number of samples per label
plt.figure(figsize=(15, 4))
ax = sns.barplot(x=classes, y=count, color='navy')
plt.xticks(rotation=285)
for i in ax.containers:
    ax.bar_label(i,)
plt.title('Number of samples per label', fontsize=25, fontweight='bold')
plt.xlabel('Labels', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.yticks(np.arange(0, 105, 10))
plt.show()

# Function to create a DataFrame for the dataset
def create_df(folder_path):
    all_images = []
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        all_images.extend([(os.path.join(class_path, file_name), class_name) for file_name in os.listdir(class_path)])
    df = pd.DataFrame(all_images, columns=['file_path', 'label'])
    return df

# Create train and test DataFrames
train_df = create_df(TRAIN_DIR)
test_df = create_df(TEST_DIR)

# Print number of samples in train and test sets
print(colored(f'Number of samples in train: {len(train_df)}', 'blue', attrs=['bold']))
print(colored(f'Number of samples in test: {len(test_df)}', 'blue', attrs=['bold']))

# Display some images from the dataset
df_unique = train_df.copy().drop_duplicates(subset=["label"]).reset_index()
fig, axes = plt.subplots(ncols=5, figsize=(8, 7), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df_unique.file_path[i]))
    ax.set_title(df_unique.label[i], fontsize=12)
plt.tight_layout(pad=0.5)
plt.show()

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    shear_range=0.1,
    fill_mode='nearest',
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='file_path',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    seed=42,
    shuffle=False
)

# Load pre-trained MobileNetV2 model without top layer
pre_trained_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze layers until block_16_expand
pre_trained_model.trainable = True
set_trainable = False
for layer in pre_trained_model.layers:
    if layer.name == 'block_16_expand':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Create the model with custom layers
model = models.Sequential()
model.add(pre_trained_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model checkpoint and early stopping
checkpoint_cb = ModelCheckpoint('MyModel.keras', save_best_only=True)
earlystop_cb = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    epochs=100,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr]
)

# Load the best model
best_model = models.load_model('MyModel.keras')

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(test_generator)
print(colored(f'Test Loss: {round(test_loss, 3)}', 'green', attrs=['bold']))
print(colored(f'Test Accuracy: {round(test_acc, 3)}', 'green', attrs=['bold']))

# Plot training results
result_df = pd.DataFrame(history.history)
x = np.arange(len(result_df))
fig, ax = plt.subplots(3, 1, figsize=(15, 12))

# Loss plot
ax[0].plot(x, result_df.loss, label='loss', linewidth=3)
ax[0].plot(x, result_df.val_loss, label='val_loss', linewidth=2, ls='-.', c='r')
ax[0].set_title('Loss', fontsize=20)
ax[0].legend()

# Accuracy plot
ax[1].plot(x, result_df.accuracy, label='accuracy', linewidth=2)
ax[1].plot(x, result_df.val_accuracy, label='val_accuracy', linewidth=2, ls='-.', c='r')
ax[1].set_title('Accuracy', fontsize=20)
ax[1].legend()

# Learning rate plot
ax[2].plot(x, result_df.lr, label='learning_rate', linewidth=2)
ax[2].set_title('Learning Rate', fontsize=20)
ax[2].set_xlabel('Epochs')
ax[2].legend()

plt.show()

# Confusion Matrix and Classification Report
def evaluate_model_performance(model, val_generator, class_labels):
    true_labels = val_generator.classes
    predictions = model.predict(val_generator, steps=len(val_generator))
    predicted_labels = np.argmax(predictions, axis=1)
    report = classification_report(true_labels, predicted_labels, target_names=class_labels)
    print(report)
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

class_labels = list(train_generator.class_indices.keys())
evaluate_model_performance(best_model, test_generator, class_labels)
