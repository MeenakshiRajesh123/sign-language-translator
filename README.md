# Sign Language Translator (AI-Powered)

A deep-learning–based **real-time Sign Language Translator** built with TensorFlow and OpenCV.  
The system recognizes static ASL (A–Z, “space”, “nothing”, “del”) signs from webcam video and translates them into text in real time.  

---

## Project Overview

This project was built step-by-step:

1. **Data Processing** – Prepared and split images into training, validation, and test folders.  
2. **Model Design & Training** – Used **MobileNetV2** with transfer learning to train on ASL dataset.  
3. **Checkpoint & Resume** – Implemented progress tracking and checkpointing for safe resuming.  
4. **Model Evaluation** – Tested model on unseen data and generated performance metrics.  
5. **Real-Time Translator** – Created a webcam-based interface for live gesture prediction.


---

## Model Architecture

- **Base Model:** `MobileNetV2` (pre-trained on ImageNet)  
- **Added Layers:**
  - `GlobalAveragePooling2D`
  - `Dense(128, activation="relu")`
  - `Dropout(0.5)`
  - `Dense(num_classes, activation="softmax")`
- **Optimizer:** `Adam (1e-4)`  
- **Loss:** `categorical_crossentropy`  
- **Metric:** `accuracy`

---

## ⚙️ Setup Instructions

### 1️. Clone the Repository

```bash
git clone https://github.com/MeenakshiRajesh123/sign-language-translator.git
cd sign-language-translator

```

### 2️. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

```
### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Train the Model

```bash
processed_dataset/train
processed_dataset/val

```
Run the training script

```bash
python src/train_model.py

```
Features:
Automatically resumes training if interrupted
Saves best model weights to: checkpoints/sign_language_model.weights.h5
Logs training progress in training_progress.json

### 5. Evaluate the Model
Run the evaluation script

```bash
python src/evaluate.py
```

Outputs:
Test accuracy and loss
Classification report (Precision, Recall, F1-score)
Confusion matrix visualization

### 6. Run Real-Time Translator

Start live predictions using your webcam
```bash
python src/realtime_translator.py
```