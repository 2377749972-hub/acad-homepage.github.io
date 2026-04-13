"""Train and evaluate LeNet on MNIST."""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy on evaluation data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train() -> None:
    # Basic training settings suitable for a normal laptop.
    batch_size = 64
    epochs = 5
    learning_rate = 0.001

    # Auto-select GPU if available; otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalize MNIST grayscale images to roughly [-1, 1].
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare log file.
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train_log.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("LeNet on MNIST training log\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {learning_rate}\n\n")

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            test_accuracy = evaluate(model, test_loader, device)

            msg = f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%"
            print(msg)
            log_file.write(msg + "\n")

    print(f"Training complete. Log saved to: {log_path}")


if __name__ == "__main__":
    train()
