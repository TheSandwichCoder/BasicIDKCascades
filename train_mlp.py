import numpy as np
from torch import nn
import torch.nn.functional as F

from globals import *
import joblib
from torch.utils.data import random_split,Dataset, DataLoader

def get_m_output(fp = ""):

    if fp == "": # temp for testing
        rng = np.random.default_rng()

        return rng.random(size = (10000, 1000))
    
    return np.load(fp)


m1_outputs = get_m_output("data/saved_runs/m1_output.npy").squeeze()
m2_outputs = get_m_output("data/saved_runs/m2_output.npy").squeeze()
m3_outputs = get_m_output("data/saved_runs/m3_output.npy").squeeze()

m1_features = get_m_output("data/saved_runs/m1_features.npy").squeeze()

def output_to_softmax(x, axis=1):
    # Subtract max for numerical stability along the specified axis
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Divide by the sum of exponentials along the same axis
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)    

# highest of prob dist
def output_to_confidence(x):
    prob = output_to_softmax(x)
    return np.max(prob, axis = 1)

# Shannon entropy
def output_to_entropy(x):
    prob = output_to_softmax(x)

    eps = 1e-12
    n = -np.sum(prob * np.log2(prob + eps), axis=1)

    return n

# difference between top 2 probabilities
def output_to_margin(x):
    prob = output_to_softmax(x)

    sorted_idx = np.argsort(prob, axis = 1)

    first_highest_key = sorted_idx[:, -1]
    second_highest_key = sorted_idx[:, -2]

    rows = np.arange(prob.shape[0])

    p1 = prob[rows, first_highest_key]
    p2 = prob[rows, second_highest_key]

    margin = p1 - p2

    return margin

def output_to_pred(x):
    return np.argmax(x, axis = 1)

def get_true_pred(pred, conf, conf_threshold):
    idk_key = conf < conf_threshold

    new_arr = pred.copy()

    new_arr[idk_key] = -1

    return new_arr

def expand_classes(y, n_classes):
    new_arr = np.zeros((y.shape[0], n_classes))

    new_arr[np.arange(y.shape[0]), y] = 1.00

    return new_arr

def expand_binary_classes(y):
    new_arr = np.zeros(y.shape[0], dtype=int)

    new_arr[y] = 1

    return expand_classes(new_arr, 2)


# DATA REFINEMENT

m1_conf = output_to_confidence(m1_outputs)
m2_conf = output_to_confidence(m2_outputs)

idk_key = m1_conf < M1_P95_CONF_THRESHOLD




labels = (m2_conf < M2_P95_CONF_THRESHOLD).astype(np.int64)

# selects only values where m1 returned idk
X = m1_features[idk_key]
y = labels[idk_key]

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        # Convert NumPy arrays directly to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 216)
        self.fc2 = nn.Linear(216, 2)
        

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

dataset = FeatureDataset(X, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset, 
    [train_size, test_size], 
    generator=generator
)

print(test_size)
train_dl = DataLoader(train_dataset)
test_dl = DataLoader(test_dataset)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# model = MLP()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dl, model, loss_fn, optimizer)
#     test(test_dl, model, loss_fn)
# print("Done!")


# torch.save(model.state_dict(), "model.pth")
# print("SAVED!")
