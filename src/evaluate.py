# src/evaluate.py

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from dataset import UrbanSoundDataset
from model import HazardSoundCNN


def evaluate():
    selected_classes = ["gun_shot", "siren", "drilling", "engine_idling", "dog_bark"]

    csv_path = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    base_path = "data/UrbanSound8K"

    dataset = UrbanSoundDataset(csv_path, base_path, selected_classes)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HazardSoundCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%\n")

    cm = confusion_matrix(all_labels, all_preds)

    print("Confusion Matrix:")
    header = " " * 15 + "".join(f"{cls:>15}" for cls in selected_classes)
    print(header)

    for i, row in enumerate(cm):
        row_str = "".join(f"{val:>15}" for val in row)
        print(f"{selected_classes[i]:>15}{row_str}")

    print("\nPer-Class Accuracy:")
    for i, cls in enumerate(selected_classes):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        acc = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"{cls}: {acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=selected_classes))


if __name__ == "__main__":
    evaluate()