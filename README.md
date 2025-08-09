---

# Vehicle Image Classification – Deep Learning Capstone with TensorFlow & Keras

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![Model Type](https://img.shields.io/badge/Model-CNN%20Binary%20Classifier-blueviolet.svg)

## Project Overview

This capstone project applies deep learning techniques to classify vehicle images into two categories: **Car** and **Bike**. Using Convolutional Neural Networks (CNNs), the goal was to build a robust image classification pipeline that could later be extended to include environmental features such as location or time of day. The project builds on previous work in bike rental analysis and introduces computer vision as a new modality for predictive modeling.

---

## Dataset Preparation

- **Source**: Manually curated image dataset split into `train/` and `validation/` folders  
- **Structure**:
  - `train/Car/` and `train/Bike/`
  - `validation/Car/` and `validation/Bike/`
- **Cleaning**:
  - Removed non-JPG formats (e.g., WEBP, PNG) to ensure compatibility
- **Final Counts**:
  - Bike (Train): 391 JPG images  
  - Car (Train): 399 JPG images

**Download Links**:  
- [train folder](https://drive.google.com/drive/folders/1Fbxf_8wN1nPCCILdfVJt93OHVw0xidG0?usp=drive_link)  
- [validation folder](https://drive.google.com/drive/folders/1np4Pbg3w8N9CKQQ8GTSiTGCdJUuCca4Q?usp=drive_link)  
- [Full dataset archive](https://drive.google.com/file/d/1sr5xheAEcvoDvIBO-b1ANjnW1GXDmn2V/view)

---

## Data Preprocessing

Used `ImageDataGenerator` for image loading and augmentation:

**Training Generator**
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

**Validation Generator**
```python
ImageDataGenerator(rescale=1./255)
```

- All images resized to 150×150 pixels  
- Batch size: 32  
- Class mode: Binary (Car vs Bike)

---

## Model Architecture

Built using TensorFlow’s Keras API:

```python
model = keras.Sequential([
    layers.Input(shape=(150,150,3)),
    layers.Flatten(),
    layers.Dense(16),
    BatchNormalization(),
    layers.ReLU(),
    layers.Dense(1, activation='sigmoid')
])
```

- Input: RGB images (150×150×3)  
- Flatten Layer: Converts image to 1D vector  
- Dense Layer: 16 units with Batch Normalization and ReLU  
- Output Layer: Sigmoid activation for binary classification

**Compilation**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
```

---

## Model Training

- Epochs: 20 (early stopping after 5 stagnant epochs)  
- Steps per epoch: Calculated manually from dataset size  
- Callback: `EarlyStopping(monitor='val_loss', patience=5)`

Training was conducted using `.fit()` with both training and validation generators.

---

## Evaluation Metrics

Final performance after 15 epochs:

| Metric              | Value   |
|---------------------|---------|
| Loss                | 0.4367  |
| Validation Accuracy | 81.01%  |
| Precision           | 79.76%  |
| Recall              | 83.75%  |

---

## Epoch Snapshots

| Epoch | Train Accuracy | Train Precision | Train Recall | Val Accuracy | Val Precision | Val Recall |
|-------|----------------|------------------|--------------|--------------|----------------|-------------|
| 1     | 67.52%         | 68.04%           | 63.84%       | 49.22%       | 100.00%        | 01.52%      |
| 5     | 78.79%         | 84.24%           | 72.68%       | 71.88%       | 87.50%         | 53.03%      |
| 10    | 81.25%         | 64.71%           | 100.00%      | 81.25%       | 79.17%         | 86.36%      |
| 15    | 78.96%         | 83.42%           | 73.04%       | 79.69%       | 80.88%         | 80.88%      |

---

## Prediction Visualization

Sample predictions on validation images:

| Prediction | Confidence | Actual Class | Result    |
|------------|------------|--------------|-----------|
| Car        | 0.57       | Bike         | Incorrect |
| Car        | 0.72       | Car          | Correct   |
| Bike       | 0.05       | Bike         | Correct   |
| Car        | 0.97       | Car          | Correct   |
| Car        | 0.96       | Car          | Correct   |

**Observations**:
- The model tends to favor the "Car" class, even with low confidence  
- Misclassifications may stem from background cues or visual similarity (e.g., motorcycles)  
- Feature extraction could be improved with convolutional layers

---

## Key Learnings & Skills Gained

### Technical Skills

- Image preprocessing with `ImageDataGenerator`  
- Manual dataset curation and format filtering  
- CNN architecture design using Keras  
- Batch Normalization and ReLU activation  
- Binary classification with sigmoid output  
- Early stopping and training optimization  
- Evaluation using accuracy, precision, recall, and F1-score  
- Visualization of predictions and confidence scores  
- Manual inspection of model behavior and misclassifications

### Conceptual Insights

- Importance of clean, well-structured image datasets  
- How augmentation improves generalization  
- Role of batch normalization in stabilizing training  
- Trade-offs between precision and recall in early epochs  
- Challenges in binary classification with visually similar classes  
- Need for convolutional layers to extract spatial features  
- Value of confidence calibration and threshold tuning

---

## Contact

**Author**: Carllos Watts-Nogueira  
**Email**: [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn**: [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---

