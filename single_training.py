import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='Directory where data is stored')
    parser.add_argument('--output-dir', type=str, help='Directory for output files')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

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

def train():
    args = parse_args()
    
    # Log the parameters using MLflow
    mlflow.log_params({
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    })
    
    # Set device (cpu vms only)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset
    print(f"downloading dataset...")
    dataset = datasets.FashionMNIST(
        root=args.data_dir, 
        train=True, 
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Create model
    model = SimpleCNN().to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # Calculate epoch statistics
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, "
            f"Time: {epoch_time:.2f} seconds")
        
        # Log metrics using MLflow instead of run.log
        mlflow.log_metric('epoch', epoch+1)
        mlflow.log_metric('train_loss', epoch_loss)
        mlflow.log_metric('train_accuracy', epoch_acc)
        mlflow.log_metric('epoch_time', epoch_time)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()