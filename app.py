import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set up the page configuration
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model(model_path="plant_disease_model.pth", num_classes=28):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Helper function to predict
def predict(image, model, class_names):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    return class_names[predicted.item()], confidence

# Load class names
@st.cache_resource
def load_class_names():
    # Ensure the class names match your dataset
    return [
        'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf',
        'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot',
        'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight',
        'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf', 'Squash Powdery mildew leaf',
        'Strawberry leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
        'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus',
        'Tomato leaf yellow virus', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf',
        'grape leaf', 'grape leaf black rot'
    ]

class_names = load_class_names()

# UI Layout
st.title("🌿 Plant Disease Detector")
st.markdown("Upload an image to detect plant diseases using a deep learning model.")

# Upload and display image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Predict the disease
        class_name, confidence = predict(image, model, class_names)
        st.subheader(f"Prediction: **{class_name}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
