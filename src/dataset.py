import os
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path, base_path, selected_classes):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

        self.df = self.df[self.df["class"].isin(selected_classes)].reset_index(drop=True)

        self.base_path = base_path
        self.selected_classes = selected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_name = str(row["slice_file_name"]).strip()
        fold = str(row["fold"]).strip()
        label_name = str(row["class"]).strip()

        file_path = os.path.normpath(
            os.path.join(self.base_path, "audio", f"fold{fold}", file_name)
        )

        try:
            signal, sr = librosa.load(file_path, sr=22050, duration=4.0)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            signal = np.zeros(22050 * 4)
            sr = 22050

        if len(signal) == 0:
            signal = np.zeros(22050 * 4)

        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, hop_length=512)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec = torch.tensor(mel_spec).float()

        if mel_spec.shape[1] < 128:
            pad = 128 - mel_spec.shape[1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
        else:
            mel_spec = mel_spec[:, :128]

        mel_spec = mel_spec.unsqueeze(0)

        label = self.class_to_idx[label_name]

        return mel_spec, label