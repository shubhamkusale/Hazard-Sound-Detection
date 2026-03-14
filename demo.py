import torch
import matplotlib.pyplot as plt
from src.dataset import UrbanSoundDataset


print("GPU Available:", torch.cuda.is_available())
print("GPU Device Count:", torch.cuda.device_count())
print("-" * 50)

selected_classes = [
    "gun_shot",
    "siren",
    "drilling",
    "engine_idling",
    "dog_bark"
]


dataset = UrbanSoundDataset(
    csv_path="data/UrbanSound8K/metadata/UrbanSound8K.csv",
    base_path="data/UrbanSound8K",
    selected_classes=selected_classes
)

print("Total Samples in Selected Classes:", len(dataset))
print("Class Mapping:", dataset.class_to_idx)
print("-" * 50)


sample, label = dataset[0]

print("Sample Tensor Shape:", sample.shape)
print("Sample Label Index:", label)
print("Label Name:", selected_classes[label])
print("-" * 50)


plt.figure(figsize=(8, 4))
plt.imshow(sample.squeeze(0), aspect='auto', origin='lower')
plt.colorbar()
plt.title("Mel Spectrogram Example")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.show()