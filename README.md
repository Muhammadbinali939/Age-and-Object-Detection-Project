# Age and Object Detection Project 

## Overview
This repository contains multiple AI-based detection projects leveraging deep learning models. The tasks include age detection, drowsiness detection, nationality classification, emotion recognition, sign language detection, and more. The repository includes Jupyter Notebooks and Python scripts for training, visualization, and real-time detection.

## Table of Contents
- [Task 1: Age Detection](#task-1-age-detection)
- [Task 2: Activation Map Visualization](#task-2-activation-map-visualization)
- [Task 3: Drowsiness Detection](#task-3-drowsiness-detection)
- [Task 4: Nationality and Emotion Detection](#task-4-nationality-and-emotion-detection)
- [Task 5: Attendance System](#task-5-attendance-system)
- [Task 6: Animal Detection](#task-6-animal-detection)
- [Task 7: Sign Language Detection](#task-7-sign-language-detection)
- [Task 8: Movie Theatre Age Restriction](#task-8-movie-theatre-age-restriction)
- [Task 9: Voice Emotion Detection](#task-9-voice-emotion-detection)
- [Task 10: Car Color Detection](#task-10-car-color-detection)

---

## Task 1: Age Detection
This model predicts the age of a person using a CNN-based deep learning model trained on a dataset of facial images.

### Files:
- `train_age_detection.ipynb`: Model training and evaluation.

### Requirements:
- TensorFlow
- Keras
- ImageDataGenerator

### Training Steps:
1. Load ResNet50 as the base model.
2. Freeze initial layers and add custom dense layers.
3. Train on the UTK dataset.
4. Save the trained model.

---

## Task 2: Activation Map Visualization
This task visualizes the activation maps of intermediate CNN layers to understand how the model processes images.

### Files:
- `visualize_activation_maps.ipynb`

### Requirements:
- TensorFlow
- Keras
- Matplotlib

### Steps:
1. Load a pre-trained VGG16 model.
2. Extract intermediate layer outputs.
3. Visualize activation maps for a given input image.

---

## Task 3: Drowsiness Detection
This project detects drowsiness using a CNN model and a real-time webcam feed.

### Files:
- `detection.ipynb`: Real-time detection using OpenCV.
- `drowziness_gui.py`: GUI for drowsiness detection (currently empty).

### Requirements:
- OpenCV
- TensorFlow
- NumPy

### Steps:
1. Capture video from the webcam.
2. Detect faces using Haar cascades.
3. Predict drowsiness using the trained model.

---

## Task 4: Nationality and Emotion Detection
This project predicts a person's nationality and emotion from facial images.

### Files:
- `nationality_detection.ipynb`: Model inference.
- `nationality_gui.py`: GUI for nationality detection (currently empty).

### Requirements:
- OpenCV
- TensorFlow
- NumPy

### Steps:
1. Load trained nationality and emotion models.
2. Predict nationality and emotion from an input image.

---

## Task 5: Attendance System
A facial recognition-based attendance system that marks attendance in a CSV file.

### Files:
- `attendance_system.ipynb`

### Requirements:
- OpenCV
- Face Recognition
- Pandas

### Steps:
1. Detect faces in a video feed.
2. Match detected faces with known faces.
3. Save attendance records in a CSV file.

---

## Task 6: Animal Detection
Detects animals in an image using a deep learning model.

### Files:
- `animal_system.ipynb`
- `animal_system.py` (currently empty)

### Requirements:
- OpenCV
- TensorFlow
- NumPy

### Steps:
1. Load a trained model.
2. Process the input image.
3. Predict animal presence.

---

## Task 7: Sign Language Detection
Detects sign language gestures in real-time.

### Files:
- `sign_language_gui.py`: Real-time sign language detection.
- `sign_language.ipynb` (currently empty)

### Requirements:
- OpenCV
- TensorFlow

### Steps:
1. Capture real-time video.
2. Predict sign language gestures.
3. Display results on the screen.

---

## Task 8: Movie Theatre Age Restriction
Restricts access to movie theatres based on detected age and emotion.

### Files:
- `movie_theatre.ipynb`

### Requirements:
- OpenCV
- TensorFlow
- Pandas

### Steps:
1. Capture video input.
2. Detect faces and predict age and emotion.
3. Restrict access if necessary.

---

## Task 9: Voice Emotion Detection
Predicts emotions from voice recordings using deep learning.

### Files:
- `voice_emotion.ipynb`
- `voice_gui.py` (currently empty)

### Requirements:
- TensorFlow
- Librosa
- NumPy

### Steps:
1. Extract MFCC features from an audio file.
2. Use a trained model to predict emotions.

---

## Task 10: Car Color Detection
Detects the color of a car from an image.

### Files:
- `car_colour.ipynb`

### Requirements:
- TensorFlow
- OpenCV
- NumPy

### Steps:
1. Load a trained car color detection model.
2. Process the input image.
3. Predict the car color.

---

## Installation
To install the dependencies, run:
```sh
pip install -r requirements.txt
```

