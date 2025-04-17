import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

# setup and cleanup for distributed training
def setup(rank, world_size):
    print(f"[Rank {rank}] Setting up process group...", flush=True)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=datetime.timedelta(seconds=60)
    )
    print(f"[Rank {rank}] Process group initialized.", flush=True)

def cleanup():
    dist.destroy_process_group()

# define a 2D CNN model
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

# function for training model
def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)

    # dataset setup
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root='data', train=True, download=False, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # model setup
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=None)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    print(f"[Rank {rank}] Starting training...", flush=True)

    for epoch in range(10):
        ddp_model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"[Rank {rank}] Batch {batch_idx}: Loss = {loss.item():.4f}", flush=True)

        epoch_time = time.time() - start_time
        print(f"[Rank {rank}] Epoch {epoch+1} completed in {epoch_time:.2f} seconds. Total Loss: {total_loss:.4f}", flush=True)

    if rank == 0:
        save_path = "model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[Rank {rank}] Model saved to {save_path}", flush=True)

    cleanup()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"[Rank {rank}] Process started with world_size={world_size}", flush=True)
    train(rank, world_size)
