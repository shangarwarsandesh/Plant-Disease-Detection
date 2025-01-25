# Title: Plant Disease Detection
## Technologies Used :
```
Data Analysis Using Python,
Computer Vision,
Deep Learning (CNN)
```

## Commands For Installing Libraries :

```
pip install numpy pandas opencv-python
pip install torch torchvision torchaudio
pip install pillow
pip install streamlit
```
## Commands For Running Project :

```
python main.py
python app.py
streamlit run app.py
```
## To Clone :
`
git clone https://github.com/shangarwarsandesh/Plant-Disease-Detection.git
`
## Dataset Links :

['Dataset Reference Link'](https://github.com/pratikkayal/PlantDoc-Dataset)


## Steps to Develop Project :
### 1.	Define the Objective 
- Clearly outline the goals of the project, e.g., identifying
specific plant diseases from leaf images or detecting general unhealthy conditions in plants.

### 2.	Data Collection 
- Gather a dataset of plant images, including healthy anddiseased plants. 
Sources: Online repositories (e.g., Kaggle, PlantVillage), field data, or smartphone images.
- Ensure diverse images with varying lighting, angles, and environments for robustness.

### 3.	Data Preprocessing 
- Label the Data: Annotate images with disease categories or "healthy" labels.
- Clean the Data: Remove duplicates, irrelevant, or low-quality images.
- Augment the Data: Apply transformations like flipping, rotation, scaling, and color variations to increase dataset size and diversity.
- Resize Images: Standardize the dimensions to reduce computational requirements.

### 4.	Model Selection 
Choose a suitable computer vision approach: 
- Pretrained Models: Use models like ResNet-18.

### 5.	Model Training 
- Split the dataset into training, validation, and test sets (e.g., 70%-20%-10%).
- Train the model using the training set, tuning hyperparameters like learning rate, batch size, and epochs.
- Validate the model periodically to check for overfitting or underfitting.
- Use techniques like early stopping and dropout to improve performance.

### 6.	Model Evaluation 
Test the model on the test dataset to evaluate metrics such as: 
- Accuracy: Overall prediction correctness.
- Precision/Recall/F1 Score: For imbalanced datasets.
- Confusion Matrix: For detailed performance on each class.

### 7.	Deployment Preparation 
Optimize the model for real-world usage: 
- Reduce size using quantization or pruning.
- Convert to formats suitable for deployment (e.g., TensorFlow Lite, ONNX).

### 8.	Build a User Interface 
Create a simple interface for end-users using Streamlit.

### 9.	Testing and Validation 
- Test the system in real-world scenarios to identify potential issues.
- Gather feedback from end-users to improve usability and accuracy.

### 10.	Deployment and Maintenance 
- Deploy the system on cloud servers, mobile apps, or local devices.
- Monitor its performance, update the model with new data, and fix bugs as needed.

### 11.	Documentation and Reporting 
- Document the project comprehensively, including methods, results, and challenges.
- Prepare a report or presentation for stakeholders.

### 12.	Future Enhancements 
- Incorporate more features, such as real-time video nalysis or multi-plant disease detection.
- Continuously update the dataset and retrain the model to improve accuracy.


## Authors :
```
1. Sandesh Shangarwar
2. Prajwal Sakhale
3. Domeshwar Thengari
4. Kunal Kakde
5. Priyank Moon
```
