import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from dataloader import read_file
from torch.utils.data import TensorDataset
import sys
print("Using Python:", sys.version)
print("Python Executable:", sys.executable)


def setup():
    """Initialize the distributed training process using torchrun."""
    dist.init_process_group(backend="nccl")  # Use "gloo" if running on CPU
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(rank)
    
    return rank, world_size

def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()

def train():
    rank, world_size = setup()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = 'ZtoNN'
    x_particles, _, y = read_file(dataset)
    print(x_particles.shape)
    training_data, validation_data = train_test_split(x_particles, test_size=0.1, shuffle=True)
    print(training_data.shape)

    # Set up distributed data loaders
    training_data_tensor = torch.tensor(training_data, dtype=torch.float32)
    validation_data_tensor = torch.tensor(validation_data, dtype=torch.float32)
    train_dataset = TensorDataset(training_data_tensor)
    val_dataset = TensorDataset(validation_data_tensor)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    training_loader = DataLoader(train_dataset, batch_size=512, sampler=train_sampler)
    validation_loader = DataLoader(val_dataset, batch_size=512, sampler=val_sampler)

    # Define model
    class ConvolutionalAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(4, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, stride=2, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 4, 3, stride=2, padding=1, output_padding=1)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = ConvolutionalAutoencoder().to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 200
    training_losses, validation_losses = [], []

    for i_epoch in range(num_epochs):
        model.train()
        training_loader.sampler.set_epoch(i_epoch)
        training_loss = 0
        for batch in tqdm(training_loader, desc=f"Rank {rank} Epoch {i_epoch}/{num_epochs}", leave=False):
            x = batch[0].to(device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(x, predictions)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(training_loader)
        training_losses.append(training_loss)

        # Validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                x = batch[0].to(device)
                predictions = model(x)
                loss = criterion(x, predictions)
                validation_loss += loss.item()
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)

        print(f"Rank {rank}: Epoch {i_epoch} - Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}")
    print("training done")
    if rank == 0:
        torch.save(model.module.state_dict(), "trained_model.pth")
        with open("training_results.pkl", "wb") as f:
            pickle.dump((training_losses, validation_losses), f)
    print("cleaning up")
    cleanup()

if __name__ == "__main__":
    train()

