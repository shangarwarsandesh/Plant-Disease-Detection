import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
input_dir = r"PlantDoc-Dataset\dataset"
output_dir = r"PlantDoc-Dataset\preprocessed-dataset"
image_size = (128, 128)  # Standard size for images
test_size = 0.2  # Proportion of test set
augmentation = True

# Create output directories
train_dir = Path(output_dir) / "train"
test_dir = Path(output_dir) / "test"
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Helper function to preprocess a single image
def preprocess_image(image_path, image_size, augment=False):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return None

    # Resize the image
    img_resized = cv2.resize(img, image_size)

    # Normalize the image
    img_normalized = img_resized / 255.0

    # Augmentation
    augmented_images = [img_normalized]
    if augment:
        # Flip horizontally
        augmented_images.append(cv2.flip(img_normalized, 1))
        # Brightness adjustments
        bright = cv2.convertScaleAbs(img_normalized, alpha=1.2, beta=30)
        augmented_images.append(bright)
        # Rotation
        h, w = img_resized.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 15, 1)  # Rotate by 15 degrees
        rotated = cv2.warpAffine(img_resized, matrix, (w, h))
        augmented_images.append(rotated)

    return augmented_images

# Process the dataset
def preprocess_dataset(input_dir, train_dir, test_dir, image_size, test_size, augment=False):
    # Collect all image paths and labels
    data = []
    labels = []
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append(file_path)
                labels.append(subdir)

    # Split dataset
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, stratify=labels)

    # Process and save images
    for subset, subset_data, subset_labels, subset_dir in [
        ("train", train_data, train_labels, train_dir),
        ("test", test_data, test_labels, test_dir),
    ]:
        for img_path, label in zip(subset_data, subset_labels):
            label_dir = Path(subset_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            augmented_images = preprocess_image(img_path, image_size, augment=(augment and subset == "train"))
            if augmented_images is None:
                continue
            for idx, img in enumerate(augmented_images):
                save_path = label_dir / f"{Path(img_path).stem}_{idx}.png"
                cv2.imwrite(str(save_path), (img * 255).astype(np.uint8))

    print(f"Preprocessing completed. Data saved to {output_dir}")

# Run the preprocessing
preprocess_dataset(input_dir, train_dir, test_dir, image_size, test_size, augmentation)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Configuration for pretrained model
model_name = "resnet18"  # Pretrained model to use
num_classes = 28  # Replace with the number of plant disease categories
batch_size = 112
learning_rate = 0.001
epochs = 20

# Directories for training and testing data
train_dir = "PlantDoc-Dataset/preprocessed-dataset/train"
test_dir = "PlantDoc-Dataset/preprocessed-dataset/test"

# Define transformations for the dataset
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load a pretrained model and modify for the dataset
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("Training complete. Model saved as 'plant_disease_model.pth'")

# Evaluation loop
def evaluate_model():
    # Load the trained model
    model.load_state_dict(torch.load("plant_disease_model.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Run training and evaluation
if __name__ == "__main__":
    train_model()
    evaluate_model()
