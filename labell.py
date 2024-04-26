import os
import random
import matplotlib.pyplot as plt
import pandas as pd

class_names = ['monkeypox', 'others']

# Function to plot images with labels
def plot_images(images, labels, predicted_labels, class_names, rows=1, figsize=(15, 10)):
    fig, axes = plt.subplots(rows, len(images)//rows, figsize=figsize)
    axes = axes.flatten()
    for i, (image, label, predicted_label) in enumerate(zip(images, labels, predicted_labels)):
        axes[i].imshow(image)
        axes[i].axis('off')
        title = f'True Label: {label}\nPredicted Label: {predicted_label}'
        axes[i].set_title(title)
    plt.tight_layout()
    plt.show()

# Function to process testing results and select random images
def process_testing_results(excel_file, test_folder):
    # Read testing results from Excel
    df = pd.read_excel(excel_file)
    
    # Select random images from each class folder
    monkeypox_images = random.sample(os.listdir(os.path.join(test_folder, 'monkeypox')), 5)
    others_images = random.sample(os.listdir(os.path.join(test_folder, 'others')), 5)
    
    # Load and plot the selected images with labels
    selected_images = [os.path.join(test_folder, 'monkeypox', img) for img in monkeypox_images] + \
                      [os.path.join(test_folder, 'others', img) for img in others_images]
    
    # True labels based on folder names
    selected_labels = ['monkeypox'] * 5 + ['others'] * 5
    
    # Generate random predicted labels for demonstration
    random_predicted_labels = random.choices(class_names, k=10)
    
    # Load images
    images = [plt.imread(img_path) for img_path in selected_images]
    
    # Plot images with labels
    plot_images(images, selected_labels, random_predicted_labels, class_names, rows=2)

# Paths
model1_excel = 'greeshma_mpox_dataset/result/vgg16/results.xlsx'
model2_excel = 'greeshma_mpox_dataset/result/xception/results.xlsx'
testing_folder = 'greeshma_mpox_dataset/dataset/Test'

# Process testing results for Model 1
process_testing_results(model1_excel, testing_folder)

# Process testing results for Model 2
process_testing_results(model2_excel, testing_folder)

print('labelled image generated successfully...')
