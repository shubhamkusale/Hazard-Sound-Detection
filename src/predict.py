# src/predict.py

import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa

from model import HazardSoundCNN

selected_classes = ["gun_shot", "siren", "drilling", "engine_idling", "dog_bark"]
HAZARD_CLASSES = ["gun_shot", "siren", "drilling"]


def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=4.0)

    target_length = 22050 * 4
    if len(y) < target_length:
        pad = target_length - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:target_length]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_tensor = torch.tensor(mel_db, dtype=torch.float32)

    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)

    mel_tensor = F.interpolate(mel_tensor, size=(128, 128), mode="bilinear", align_corners=False)

    mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)

    return mel_tensor


def predict_sound(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HazardSoundCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    spectrogram = preprocess_audio(audio_path).to(device)

    with torch.no_grad():
        outputs = model(spectrogram)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = selected_classes[pred.item()]
    confidence = conf.item() * 100

    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    if predicted_class in HAZARD_CLASSES:
        print("⚠ HAZARD DETECTED")
    else:
        print("✓ Safe sound")

    return predicted_class, confidence


if __name__ == "__main__":
    predict_sound(sys.argv[1])