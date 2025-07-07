from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import wnae
from wnae.wnae import WNAE
import numpy as np
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torch.utils.data import TensorDataset
import sys
from computeAxes import compute_axes_torch
import argparse

parser = argparse.ArgumentParser(description = "Autoencoder for Jet Data")
parser.add_argument("training_dataset", type = str, default = 'WtoQQScaled.npy', help = "Typically WtoQQ, the dataset the model is trained on")
parser.add_argument("test_dataset", type = str, default = 'TTBarScaled.npy', help = "What Jet the model is trained on")
parser.add_argument("results", type = str, default = 'results', help = "Where losses are saved to")
parser.add_argument("weights", type = str, default = 'weights', help = "Where model weights are saved to")
parser.add_argument("epochs", type = int, default = 300, help = "How many epochs the model should run for")
args=parser.parse_args()

x = np.load(args.training_dataset)
t = np.load(args.test_dataset)

# Split data
training_data, validation_data = train_test_split(x, test_size=0.2, shuffle=True)


device = 'cuda'

def get_loader(data):
    data = torch.tensor(data.astype(np.float32)).to(device)
    sampler = torch.utils.data.RandomSampler(
        data_source=data,
        num_samples=2**17,
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(data),
        batch_size=512,
        sampler=sampler,
    )

    return loader

training_loader = get_loader(training_data)
validation_loader = get_loader(validation_data)
test_loader = get_loader(t)



class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 9)
        self.layer2 = nn.Linear(9, 6)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.layer1 = nn.Linear(6, 9)
        self.layer2 = nn.Linear(9, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x

wnae_parameters = {
    "sampling": "pcd",
    "n_steps": 12,
    "step_size": None,
    "noise": 0.1,
    "temperature": 0.05,
    "bounds": (-4, 4),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.95,
    "buffer_size": 10000,
}

model = WNAE(
    encoder=Encoder(input_size=9),
    decoder=Decoder(output_size=9),
    **wnae_parameters,
)

model.to(device)
for name, param in model.named_parameters():
    print(f"{name} on device: {param.device}")
def run_training(model, loss_function, n_epochs, plot_epochs):

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3e-4,
    )

    training_losses = []
    validation_losses = []
    mcmc_samples_list = []
    test_losses = []
    negative_samples = []
    positive_samples = []

    for i_epoch in range(n_epochs):

        # Train step
        model.train()
        n_batches = 0
        training_loss = 0
        bar_format = f"Epoch {i_epoch}/{n_epochs}: " \
            + "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        for batch in tqdm(training_loader, bar_format=bar_format):
            n_batches += 1
            x = batch[0]
            optimizer.zero_grad()
            # Use the `train_step` method to compute the loss
            if loss_function == "wnae":
                loss, training_dict = model.train_step(x)
            elif loss_function == "nae":
                loss, training_dict = model.train_step_nae(x)
            elif loss_function == "ae":
                loss, training_dict = model.train_step_ae(x, run_mcmc=True, mcmc_replay=True)
            loss.backward()
            optimizer.step()

            training_loss += training_dict["loss"]
            if n_batches == 1 and i_epoch in plot_epochs:
                negative_samples.append(training_dict['negative_energy'])
                positive_samples.append(training_dict['positive_energy'])
           
        training_loss /= n_batches
        training_losses.append(training_loss)

    # Validation step
        model.eval()
        n_batches = 0
        validation_loss = 0
        for batch in validation_loader:
            n_batches += 1
            x = batch[0]
        # Use the `validation_step` method to get the loss without
        # changing the internal state of the model
            if loss_function == "wnae":
                validation_dict = model.validation_step(x)
            elif loss_function == "nae":
                validation_dict = model.validation_step_nae(x)
            elif loss_function == "ae":
                validation_dict = model.validation_step_ae(x, run_mcmc=True)
            validation_loss += validation_dict["loss"]
            # Only store the MCMC samples for visualization purpose for a few batches
            if n_batches == 1 and i_epoch in plot_epochs:
                mcmc_samples_list.append(validation_dict["mcmc_data"]["samples"][-1])

        validation_loss /= n_batches
        validation_losses.append(validation_loss)

        n_batches = 0
        test_loss = 0
        for batch in test_loader:
            n_batches += 1
            x = batch[0]
        # Use the `validation_step` method to get the loss without
        # changing the internal state of the model
            if loss_function == "wnae":
                validation_dict = model.validation_step(x)
            elif loss_function == "nae":
                validation_dict = model.validation_step_nae(x)
            elif loss_function == "ae":
                validation_dict = model.validation_step_ae(x, run_mcmc=True)
            test_loss += validation_dict["loss"]

        test_loss /= n_batches
        test_losses.append(test_loss)



    return training_losses, validation_losses, test_losses, mcmc_samples_list, positive_samples, negative_samples


plot_epochs = [0, 10, 50, 100, 150, 175, 200, 225, 250, 300]
n_epochs = args.epochs

training_losses, validation_losses, test_losses, mcmc_samples_list, positive_samples, negative_samples = \
run_training(model, "wnae", n_epochs, plot_epochs)

print("training done")



with open(args.results + '.pkl', "wb") as f:
     pickle.dump((training_losses, validation_losses, test_losses, positive_samples, negative_samples), f)


torch.save(model.state_dict(), args.weights + '.pth')

    
print("cleaning up")


