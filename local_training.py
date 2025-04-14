import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import time

# setup and cleanup for distributed training
def setup(rank, world_size):
    print(f"[Rank {rank}] Setting up process group... (before)")
    try:
        dist.init_process_group(
            backend="gloo",
            init_method='tcp://20.81.150.5:29501',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        print(f"[Rank {rank}] Process group initialized. (after)")
    except Exception as e:
        print(f"[Rank {rank}] Error during init_process_group: {e}")

def cleanup():
    dist.destroy_process_group()

# defining model
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

# training model
def train(rank, world_size):
    if rank == 1:
        time.sleep(10)
    setup(rank, world_size)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)

    # dataset and dataloader
    transform = transforms.ToTensor()
    download = True if rank == 0 else False
    dataset = datasets.FashionMNIST(root='data', train=True, download=download, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # model, loss, optimizer
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=None)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    # loop for training
    print(f"[Rank {rank}] Starting training...")
    for epoch in range(1):
        ddp_model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"[Rank {rank}] Batch {batch_idx}: Loss = {loss.item():.4f}")

        print(f"[Rank {rank}] Epoch {epoch+1} completed. Total Loss: {total_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    import sys
    rank = int(sys.argv[1])  # 0 or 1
    world_size = 2
    train(rank, world_size)
