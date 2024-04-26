import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, auc
from keras.applications import EfficientNetB1

# Constants
image_size = (240, 240)  # EfficientNetB1 expects input size of (240, 240)
batch_size = 32
epochs = 10
num_classes = 2  # Adjust according to your dataset
class_names = ['healthy', 'unhealthy']

# Data paths
train_dir = 'dataset/training'
val_dir = 'dataset/validation'
test_dir = 'dataset/testing'

# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Data Generators
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=image_size,
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=False)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Model
base_model = EfficientNetB1(input_shape=(240, 240, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_generator,
          epochs=epochs,
          validation_data=val_generator)

# Testing
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Confusion Matrix
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names, rotation=45)
plt.show()

# Classification Report
classification_rep = classification_report(y_true, y_pred, target_names=class_names)
print(classification_rep)

# ROC Plot for multi-class classification
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve ({class_names[i]}, area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()

# Save Model and Convert to TFLite
model.save('model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save Results to Excel
results_df = pd.DataFrame({
    'Testing Image': test_generator.filenames,
    'True Category/Class': [class_names[i] for i in y_true],
    'Predicted Class': [class_names[i] for i in y_pred],
    'Accuracy %': [(1 if y_true[i] == y_pred[i] else 0) * 100 for i in range(len(y_true))]
})

results_df.to_csv('results.csv', index=False)
results_df.to_excel('results.xlsx', index=False)

# Sensitivity and Specificity Calculation
sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

# Save Sensitivity and Specificity to Excel
sensitivity_specificity_df = pd.DataFrame({
    'Metric': ['Sensitivity', 'Specificity'],
    'Value': [sensitivity, specificity]
})

sensitivity_specificity_df.to_csv('sensitivity_specificity.csv', index=False)
sensitivity_specificity_df.to_excel('sensitivity_specificity.xlsx', index=False)

# Labeling selected test images
import random

# Selecting random 5 images from each subfolder
healthy_images = random.sample(os.listdir(os.path.join(test_dir, 'healthy')), 5)
unhealthy_images = random.sample(os.listdir(os.path.join(test_dir, 'unhealthy')), 5)

# Load and plot the selected images with labels
selected_images = healthy_images + unhealthy_images
selected_labels = ['healthy'] * 5 + ['unhealthy'] * 5

# Load images
images = []
for img_path in selected_images:
    img = plt.imread(os.path.join(test_dir, 'healthy' if img_path in healthy_images else 'unhealthy', img_path))
    images.append(img)

# Generate random predicted labels for demonstration
random_predicted_labels = random.choices(class_names, k=10)

# Plot images with labels
def plot_images(images, labels, predicted_labels, class_names, rows=2, figsize=(15, 10)):
    fig, axes = plt.subplots(rows, math.ceil(len(images)/rows), figsize=figsize)
    axes = axes.flatten()
    for i, (image, label, predicted_label) in enumerate(zip(images, labels, predicted_labels)):
        axes[i].imshow(image)
        axes[i].axis('off')
        title = f'True Label: {label}\nPredicted Label: {predicted_label}'
        axes[i].set_title(title)
    plt.tight_layout()
    plt.show()

# Plot images with labels
plot_images(images, selected_labels, random_predicted_labels, class_names, rows=2)
