import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import mlflow

# Setup distributed training
def setup():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        print(f"[Rank {rank}] Initializing process group...", flush=True)
        dist.init_process_group(backend="gloo")  # Use "gloo" for CPU
        print(f"[Rank {rank}] Process group initialized.", flush=True)

    return rank, world_size, local_rank

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# Define CNN model
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

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()

# Main training logic
def train():
    args = parse_args()
    rank, world_size, local_rank = setup()

    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}] Using device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None)
    )

    model = SimpleCNN().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if rank == 0:
        with mlflow.start_run():
            mlflow.log_params({
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            })

            for epoch in range(args.epochs):
                model.train()
                if sampler:
                    sampler.set_epoch(epoch)

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
                        print(f"[Rank {rank}] Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}", flush=True)

                epoch_loss = total_loss / len(dataloader)
                epoch_acc = 100.0 * correct / total
                epoch_time = time.time() - start_time

                print(f"[Rank {rank}] Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s", flush=True)

                mlflow.log_metric("train_loss", epoch_loss, step=epoch+1)
                mlflow.log_metric("train_accuracy", epoch_acc, step=epoch+1)
                mlflow.log_metric("epoch_time", epoch_time, step=epoch+1)

            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "model.pth")
            torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            print(f"[Rank {rank}] Model saved to {save_path}", flush=True)
    else:
        # Workers still need to train even if they don't log
        for epoch in range(args.epochs):
            model.train()
            if sampler:
                sampler.set_epoch(epoch)
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    cleanup()


if __name__ == "__main__":
    train()
