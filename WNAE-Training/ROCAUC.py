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
parser.add_argument("results", type = str, default = 'results', help = "Where AUC and ROC is saved to")
parser.add_argument("weights", type = str, help = "The model to be evaluated")
args=parser.parse_args()

x = np.load(args.training_dataset)
t = np.load(args.test_dataset)

# Split data


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
        batch_size=4,
        sampler=sampler,
    )

    return loader

training_loader = get_loader(x)

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

model.load_state_dict(torch.load(args.weights + '.pth'))

model.to(device)

BackGroundLosses = []
SignalLosses = []
model.eval()

validationloss = 0
for batch in tqdm(training_loader):
    x = batch[0]
    validation_dict = model.validation_step(x)
    BackGroundLosses.append(validation_dict['loss'])

testloss = 0
for batch in tqdm(test_loader):
    y = batch[0]
    validation_dict = model.validation_step(y)
    SignalLosses.append(validation_dict['loss'])



data = np.array(BackGroundLosses)
background = (data - data.min()) / (data.max() - data.min())


data = np.array(SignalLosses)
signal = (data - data.min()) / (data.max() - data.min())
from sklearn.metrics import roc_curve, auc
# Convert to arrays first
background_preds = np.array(background)
signal_preds = np.array(signal)

# Concatenate predictions and labels
y_scores = np.concatenate([background_preds, signal_preds])
y_true = np.concatenate([[0] * len(background_preds), [1] * len(signal_preds)])

# Combined prediction scores

# Convert to NumPy arrays (optional but standard)
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)




with open(args.results + '.pkl', "wb") as f:
     pickle.dump((fpr, tpr, roc_auc, thresholds), f)




    
print("cleaning up")


