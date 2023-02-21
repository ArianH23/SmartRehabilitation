import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
from torch import topk
import pandas as pd
from collections.abc import Iterable
from torch.utils.data import Dataset, DataLoader
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

df = pd.read_csv('../dataset_dist_to_min/data.csv')
df = df.drop('col64', axis=1)

le = preprocessing.LabelEncoder()
le.fit(['power', 'precision'])


df['col65'] = le.transform(df['col65'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True )

train_data = Data(np.array(X_train), np.array(y_train))
train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

test_data = Data(np.array(X_val), np.array(y_val))
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

input_dim = 63
hidden_dim_1 = 32
hidden_dim_2 = 16
output_dim = 1

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim_1)
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.layer_3 = nn.Linear(hidden_dim_2, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.sigmoid(self.layer_3(x))

        return x
       
model = Net(input_dim, hidden_dim_1, hidden_dim_2, output_dim)

learning_rate = 0.01

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


num_epochs = 50
loss_values = []


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        print(loss)


print("Training Complete")

step = np.linspace(0, 100, 1980*5)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


y_pred = []
y_test = []

total = 0
correct = 0

with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()

print(f'Accuracy of the network on the {len(X_val)} test instances: {100 * correct // total}%')



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns


y_pred = list(itertools.chain(*y_pred))
y_test = list(itertools.chain(*y_test))


print(classification_report(y_test, y_pred))


cf_matrix = confusion_matrix(y_test, y_pred)

plt.subplots(figsize=(8, 5))

sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

plt.show()