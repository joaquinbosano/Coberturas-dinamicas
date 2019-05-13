#import packages

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the dataset

datos = "base.pkl"
futures = pickle.load(open(datos, "rb"))

futures.head()

my_tuple = pd.DataFrame(futures).shape
num_contracts = my_tuple[1]
num_days = my_tuple[0]

# Divide the dataset into training and evaluation

futures.iloc[1638:,:].head()

futures_train = futures.iloc[:1638,:]
futures_eval = futures.iloc[1638:,:]


x_eval = torch.from_numpy(np.array(futures_eval)).float()
x_train = torch.from_numpy(np.array(futures_train)).float()

# Setup the neural nets

class NNavg(nn.Module):


    def __init__(self):

        super().__init__()
        self.out = nn.Sequential(
        nn.Linear(7,2),
        nn.PReLU(),
        nn.Linear(2,6),
        nn.SELU(),
        nn.Linear(6,2),
        nn.ELU(),
        nn.Linear(2,5),
        nn.Hardshrink(),
        nn.Linear(5,2),
        nn.Tanh(),
        nn.Linear(2,2)
                )
    def forward(self,x):

        return self.out(x)

neural_means = NNavg()

optimizer_means = optim.Adam(neural_means.parameters(), lr=0.01)


class NNcov1(nn.Module):

    def __init__(self):

        super().__init__()
        self.out = nn.Sequential(
        nn.Linear(7,2),
        nn.PReLU(),
        nn.Linear(2,6),
        nn.SELU(),
        nn.Linear(6,2),
        nn.ELU(),
        nn.Linear(2,5),
        nn.Hardshrink(),
        nn.Linear(5,2),
        nn.Tanh(),
        nn.Linear(2,2)
                )
    def forward(self,x):

        return self.out(x)

neural_covs1 = NNcov1()

optimizer_cov1 = optim.Adam(neural_covs1.parameters(), lr=0.01)


class NNcov2(nn.Module):

    def __init__(self):

        super().__init__()
        self.out = nn.Sequential(
        nn.Linear(7,2),
        nn.PReLU(),
        nn.Linear(2,6),
        nn.SELU(),
        nn.Linear(6,2),
        nn.ELU(),
        nn.Linear(2,5),
        nn.Hardshrink(),
        nn.Linear(5,2),
        nn.Tanh(),
        nn.Linear(2,2)
                )
    def forward(self,x):

        return self.out(x)


neural_covs2 = NNcov2()
optimizer_cov2 = optim.Adam(neural_covs2.parameters(), lr=0.01)


state = torch.from_numpy(np.load("state_means_SM.npy")).float()

means_train = state[:1638, :]
means_eval = state[1638:,:]

optimizer = optim.Adam(neural_covs1.parameters(), lr=0.01)
criterion = nn.MSELoss()

cov = torch.from_numpy(np.load("cov_estados_crudo_SM.npy")).float()

cov_train = cov[:1638, :, :]
cov_eval = cov[1638:,:,:]



#Covariance Training

cov_train.size()
y_cov1 = cov_train[::,0]
y_cov2 =  cov_train[::,1]


# Train step

cov_1_train_loss = []

for t in range(1000):

    neural_covs1.train()
    optimizer_cov1.zero_grad()
    y_ = neural_covs1(x_train)
    loss = criterion(y_, y_cov1)
    loss.backward()
    optimizer_cov1.step()
    cov_1_train_loss.append(loss.item())
    print (t, loss.item())
    neural_covs1.eval()

    with torch.no_grad():
        y_ = neural_covs1(x_train)

cov_2_train_loss = []

for t in range(1000):

    neural_covs2.train()
    optimizer_cov2.zero_grad()
    y_ = neural_covs2(x_train)
    loss = criterion(y_, y_cov2)
    cov_2_train_loss.append(loss.item())
    loss.backward()
    optimizer_cov2.step()
    print (t, loss.item())
    neural_covs2.eval()

    with torch.no_grad():
        y_ = neural_covs2(x_train)

# Means training

means_loss = []
for t in range(1000):

    neural_means.train()
    optimizer_means.zero_grad()
    y_ = neural_means(x_train)
    loss = criterion(y_, means_train)
    means_loss.append(loss.item())
    loss.backward()
    optimizer_means.step()
    perdida.append(loss.item())
    print (t, loss.item())
    neural_means.eval()

    with torch.no_grad():
        y_ = neural_means(x_train)


## If you want to print the model parameters

for param in neural_covs1.parameters():
    print (param)

for param in neural_covs2.parameters():
    print (param)

for param in neural_means.parameters():
    print (param)


# Train Loss Covariance
t = np.arange(0, 1000)

fig, ax = plt.subplots(figsize=(15, 9))
plt.plot(t, cov_1_train_loss )

plt.xlabel("Time (Epochs)",  fontsize = "x-large")
plt.ylabel('First State Covariance Quadratic Training Loss',  fontsize = "x-large")

plt.savefig("covariance_training_loss1.png")
plt.show()

cov_1_train_loss[-1]

fig, ax = plt.subplots(figsize=(15, 9))
plt.plot(t, cov_2_train_loss )

plt.xlabel("Time (Epochs)",  fontsize = "x-large")
plt.ylabel('Second State Covariance Quadratic Training Loss',  fontsize = "x-large")

plt.savefig("covariance_training_loss2.png")
plt.show()

cov_2_train_loss[-1]

# Train Loss Means

fig, ax = plt.subplots(figsize=(15, 9))
plt.plot(t, means_loss)

plt.xlabel("Time",  fontsize = "x-large")
plt.ylabel('State Means Quadratic Loss',  fontsize = "x-large")

plt.savefig("means_training_loss.png")
plt.show()

means_loss[-1]
### Evaluation Loss Covariance

y_cov1_eval = cov_eval[::,0]

neural_covs1.eval()
evaluation_loss_cov1 = []

with torch.no_grad():
    x_pred = neural_covs1.forward(x_eval)



for value in range(x_eval.shape[0]):

    loss2 = criterion(y_cov1_eval[value],x_pred[value,:])
    evaluation_loss_cov1.append(loss2)


t_eval = np.arange(x_eval.shape[0])

fig, ax = plt.subplots(figsize=(15, 9))
plt.plot(t_eval, evaluation_loss_cov1)

plt.xlabel("Time (Trading Days)",  fontsize = "x-large")
plt.ylabel('Covariance Evaluation Loss',  fontsize = "x-large")

plt.savefig("Covariance_evaluation_loss_1.png")
plt.show()

########################################

y_cov2_eval = cov_eval[::,1]

neural_covs1.eval()
evaluation_loss_cov2 = []


neural_covs2.eval()
with torch.no_grad():
    x_pred = neural_covs2.forward(x_eval)

for value in range(x_eval.shape[0]):
    loss2 = criterion(y_cov2_eval[value],x_pred[value])
    evaluation_loss_cov2.append(loss2)

t_eval = np.arange(x_eval.shape[0])

fig, ax = plt.subplots(figsize=(15, 9))
plt.plot(t_eval, evaluation_loss_cov2)

plt.xlabel("Time (Trading Days)",  fontsize = "x-large")
plt.ylabel('Covariance Evaluation Loss',  fontsize = "x-large")

plt.savefig("Covariance_evaluation_loss_2.png")
plt.show()

### Evaluation Loss Means

neural_means.eval()
evaluation_loss_means = []


with torch.no_grad():
    x_pred = neural_means.forward(x_eval)

for value in range(x_pred.shape[0]):
    loss2 = criterion(means_eval[value],x_pred[value])
    evaluation_loss_means.append(loss2)

t_eval = np.arange(x_pred.shape[0])

fig, ax = plt.subplots(figsize=(15, 9))
plt.plot(t_eval, evaluation_loss_means)

plt.xlabel("Time",  fontsize = "x-large")
plt.ylabel('Means Evaluation Loss',  fontsize = "x-large")

plt.savefig("Means_evaluation_loss.png")
plt.show()

#Mean Evaluation Loss for the entire dataset

np.mean(evaluation_loss_cov1)
np.mean(evaluation_loss_cov2)
np.mean(evaluation_loss_means)
