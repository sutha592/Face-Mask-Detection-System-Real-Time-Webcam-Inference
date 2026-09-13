# Face Mask Detection System - Real-Time Webcam Inference

A real-time **Face Mask Detection System** built using **Python, OpenCV, TensorFlow/Keras, and MobileNetV2**. The system detects human faces from a live webcam feed and classifies whether each detected person is **wearing a face mask or not**.

## 📌 Project Overview

This project uses a two-stage computer vision pipeline:

1. **Face Detection** – Detects faces in each webcam frame using OpenCV's deep learning-based SSD face detector.
2. **Mask Classification** – Classifies each detected face as:

   * 😷 **Mask**
   * ❌ **No Mask**

The classification model is built using **MobileNetV2 Transfer Learning**, making the system lightweight and suitable for real-time inference.

## ✨ Features

* Real-time face detection using webcam
* Face mask / no-mask classification
* Multiple face detection in a single frame
* Confidence score for every prediction
* Bounding boxes around detected faces
* Real-time inference using OpenCV
* Transfer learning with MobileNetV2
* Data augmentation during model training
* Model evaluation using classification metrics
* Training accuracy and loss visualization

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **TensorFlow / Keras**
* **MobileNetV2**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **imutils**

## 🧠 Machine Learning Approach

### Face Detection

The project uses OpenCV's pre-trained **SSD (Single Shot Detector)** model with the ResNet-10 architecture to locate faces in webcam frames.

The face detector uses:

* `deploy.prototxt`
* `res10_300x300_ssd_iter_140000.caffemodel`

### Face Mask Classification

The mask classifier is developed using **MobileNetV2** with ImageNet pre-trained weights.

The classification head consists of:

* Average Pooling
* Flatten layer
* Dense layer with 128 neurons
* Dropout layer
* Softmax output layer with 2 classes

### Classes

The model predicts two categories:

```text
with_mask
without_mask
```

## 📂 Project Structure

```text
Face-Mask-Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── detect_mask_video.py
├── train_mask_detector.py
├── mask_detector.model
├── plot.png
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Face-Mask-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The original project uses older TensorFlow/Keras dependencies. If you encounter compatibility issues on a newer Python version, use a compatible Python environment or update the dependency versions accordingly.

## 🚀 Running the Project

The trained model is already included as:

```text
mask_detector.model
```

To start real-time mask detection:

```bash
python detect_mask_video.py
```

The system will open the webcam and display the detected faces with their prediction and confidence score.

Example:

```text
Mask: 98.45%
No Mask: 96.21%
```

Press:

```text
q
```

to exit the webcam window.

## 🏋️ Training the Model

If you want to train the mask detection model from the dataset, run:

```bash
python train_mask_detector.py
```

The training script:

1. Loads images from the dataset.
2. Resizes images to `224 × 224`.
3. Applies MobileNetV2 preprocessing.
4. Splits the dataset into training and testing sets.
5. Applies image augmentation.
6. Loads MobileNetV2 with ImageNet weights.
7. Adds a custom classification head.
8. Trains the classifier for 20 epochs.
9. Evaluates the model using a classification report.
10. Saves the trained model as `mask_detector.model`.
11. Generates a training performance plot as `plot.png`.

## 🔄 System Workflow

```text
Webcam
   │
   ▼
Capture Video Frame
   │
   ▼
Face Detection
(OpenCV SSD + ResNet-10)
   │
   ▼
Extract Face Region
   │
   ▼
Resize to 224 × 224
   │
   ▼
MobileNetV2 Preprocessing
   │
   ▼
Mask Classification
   │
   ├───────────────┐
   ▼               ▼
 With Mask      Without Mask
   │               │
   └───────┬───────┘
           ▼
Display Label + Confidence
```

## 📊 Model Training

The model uses **transfer learning** instead of training a CNN completely from scratch.

### Training Configuration

| Parameter        | Value               |
| ---------------- | ------------------- |
| Base Model       | MobileNetV2         |
| Input Size       | 224 × 224 × 3       |
| Learning Rate    | 0.0001              |
| Epochs           | 20                  |
| Batch Size       | 32                  |
| Optimizer        | Adam                |
| Loss Function    | Binary Crossentropy |
| Classes          | 2                   |
| Train/Test Split | 80/20               |

### Data Augmentation

The training pipeline applies:

* Rotation
* Zoom
* Width shifting
* Height shifting
* Shearing
* Horizontal flipping

This helps the model generalize better to different face positions and orientations.

## 📈 Evaluation

After training, the model generates a classification report containing performance metrics such as:

* Precision
* Recall
* F1-score
* Support

Training and validation **loss and accuracy** are also plotted and saved as:

```text
plot.png
```

## 💡 How It Works

For every frame captured from the webcam:

1. The frame is resized for faster processing.
2. The SSD face detector identifies faces.
3. Low-confidence detections are filtered out.
4. Each detected face is cropped from the frame.
5. The face image is converted from BGR to RGB.
6. The image is resized to `224 × 224`.
7. MobileNetV2 preprocessing is applied.
8. The trained mask classifier predicts the probability of both classes.
9. The class with the highest probability is selected.
10. A bounding box, label, and confidence percentage are displayed.

## 🎯 Applications

This system can be used as a foundation for:

* Workplace safety monitoring
* Public-space monitoring systems
* Educational institution safety systems
* Healthcare environments
* Smart surveillance applications
* Computer vision learning projects

## 🔮 Future Enhancements

Possible improvements include:

* Add a web-based interface using Flask or Streamlit
* Support image and video file input
* Add face tracking for smoother predictions
* Improve model accuracy with a larger dataset
* Add mask-type classification
* Store detection logs
* Add alert notifications for no-mask detection
* Deploy the model as a cloud application
* Optimize inference using TensorFlow Lite
* Add dashboard-based monitoring

## ⚠️ Limitations

* Detection performance depends on webcam quality and lighting conditions.
* Very small, partially visible, or heavily occluded faces may not be detected correctly.
* Model predictions may vary depending on the training dataset.
* Real-time performance depends on available CPU/GPU resources.
* The included training script contains a machine-specific dataset path that should be updated before retraining.

## 👨‍💻 Author

**Sutha S**

Final Year Engineering Student

---

## 📄 License

This project is intended for educational and learning purposes. Please verify the licenses of the datasets and pre-trained models before using the project commercially.
