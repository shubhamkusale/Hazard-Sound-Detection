# src/train.py

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from dataset import UrbanSoundDataset
from model import HazardSoundCNN


def train():
    selected_classes = ["gun_shot", "siren", "drilling", "engine_idling", "dog_bark"]

    csv_path = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    base_path = "data/UrbanSound8K"

    dataset = UrbanSoundDataset(csv_path, base_path, selected_classes)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HazardSoundCNN(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    torch.save(model.state_dict(), "model.pth")

    print("Training complete. Model saved to model.pth")


if __name__ == "__main__":
    train()