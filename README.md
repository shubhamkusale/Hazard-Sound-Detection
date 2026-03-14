# 🚨 Hazard Sound Detection System

A deep learning system that detects dangerous environmental sounds using **PyTorch** and **audio signal processing**.

This project classifies sounds such as **gunshots, sirens, drilling, engine idling, and dog barking** from audio recordings.

The system converts audio into **Mel-Spectrograms** and uses a **Convolutional Neural Network (CNN)** to classify the sound.

---

# 📌 Features

* Environmental sound classification
* Deep CNN model built with **PyTorch**
* Mel-Spectrogram audio feature extraction
* Training pipeline
* Evaluation with confusion matrix and classification report
* Sound prediction from audio file
* Hazard alert system

---

# 🧠 Classes Detected

| Class         | Hazard   |
| ------------- | -------- |
| gun_shot      | ⚠ Hazard |
| siren         | ⚠ Hazard |
| drilling      | ⚠ Hazard |
| engine_idling | Safe     |
| dog_bark      | Safe     |

If a hazard sound is detected, the system triggers a **terminal alert**.

---

# 🏗 Project Structure

```
Hazard-Sound-Detection
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── alert.py
│   └── process.py
│
├── demo.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Model Architecture

Input: **Mel Spectrogram (1 × 128 × 128)**

```
Conv2D (1 → 32)
BatchNorm
ReLU
MaxPool

Conv2D (32 → 64)
BatchNorm
ReLU
MaxPool

Conv2D (64 → 128)
BatchNorm
ReLU
AdaptiveAvgPool

Fully Connected (128 → 256)
Dropout

Output Layer (256 → 5 classes)
```

---

# 📂 Dataset

This project uses the **UrbanSound8K dataset**.

Download it from:

https://urbansounddataset.weebly.com/

After downloading, extract it to:

```
data/UrbanSound8K/
```

Expected structure:

```
data/
 └── UrbanSound8K
      ├── audio
      └── metadata
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/shubhamkusale/Hazard-Sound-Detection.git
cd Hazard-Sound-Detection
```

Create virtual environment:

```
python -m venv .venv
```

Activate environment:

Windows:

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

```
python src/train.py
```

This will train the CNN and save the model:

```
model.pth
```

---

# 📈 Evaluate the Model

```
python src/evaluate.py
```

This prints:

* Accuracy
* Confusion Matrix
* Classification Report

---

# 🔎 Predict Sound from Audio

```
python src/predict.py path/to/audio.wav
```

Example output:

```
Predicted: gun_shot
Confidence: 94%
⚠ HAZARD DETECTED
```

---

# 🚨 Run Hazard Alert System

```
python src/alert.py path/to/audio.wav
```

Example:

```
╔══════════════════════════════╗
║ ⚠ HAZARD DETECTED            ║
║ Sound: gun_shot              ║
║ Confidence: 94%              ║
╚══════════════════════════════╝
```

---

# 🛠 Technologies Used

* Python
* PyTorch
* Librosa
* NumPy
* Scikit-learn

---

# 🎯 Learning Goals

This project demonstrates:

* Audio signal processing
* Mel Spectrogram feature extrac
