import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define a simple 2D convolutional neural network
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

# create instance of model, loss function, optimizer to update model's weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model (loop once just for testing)
for epoch in range(1):
    model.train()
    total_loss = 0
    for images, labels in train_data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch done. Loss: {total_loss:.4f}")

# save the model
torch.save(model.state_dict(), "simpleCNN.pth")
print("Model saved as simpleCNN.pth")
