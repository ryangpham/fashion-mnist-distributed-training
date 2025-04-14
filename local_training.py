import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# define a simple 2D CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# load dataset
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# model, device, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(1):
    model.train()
    total_loss = 0
    num_batches = len(train_data_loader)

    print(f"\n[Epoch {epoch+1}] Training started...")

    batch_start_time = time.time()
    for batch_idx, (images, labels) in enumerate(train_data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # print batch info every 100 batches
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - batch_start_time
            print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f} - Time: {elapsed:.2f}s")
            batch_start_time = time.time()

epoch_time = time.time() - start_time
print(f"\nEpoch completed in {epoch_time:.2f} seconds. Total loss: {total_loss:.4f}")

# save model
torch.save(model.state_dict(), "simpleCNN.pth")
print("Model saved as simpleCNN.pth")
