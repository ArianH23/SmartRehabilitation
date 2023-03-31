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
import tqdm
import copy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import seaborn as sns
# imports from captum library
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from sklearn.metrics import f1_score
from collections import Counter

normalize = False
n_epochs = 30

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

df = pd.read_csv('data/depths_added.csv')

df = df.drop('picture_name', axis=1)

# le = preprocessing.LabelEncoder()
# le.fit(['none', 'power', 'precision'])

max_dist = df['lh_depth_dist'].abs().max() if df['lh_depth_dist'].abs().max() > df['rh_depth_dist'].abs().max() else df['rh_depth_dist'].abs().max()

def apply_func(row):
    for i in range(0, 96):
        if i % 2 == 0:
            row[i] = row[i] / 1920
        else:
            row[i] = row[i] / 1080
    
    row[96] = row[96] / max_dist
    row[97] = row[97] / max_dist
    
    return row

if normalize:
    df = df.apply(apply_func, axis=1)
        
# df['col65'] = le.transform(df['col65'])

print(df)
batch_size = 32
# Tips of the fingers positions
#   |        X           |           Y          |            Z          |
l = [0, 4, 8, 12, 16, 20, 21, 25, 29, 33, 37, 41]# 42, 46, 50, 54, 58, 62]

# df['col65'] = le.transform(df['col65'])

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1:]
# y = np.array(df).reshape(-1, 1)

print(y.value_counts())
# print(y)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# print(ohe.categories_)

y = ohe.transform(y)
print(ohe.categories_)

# train_data = Data(np.array(X_train), np.array(y_train))
# train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


input_dim = 98
hidden_dim_1 = 64
hidden_dim_2 = 32
output_dim = 3
# batches_per_epoch = len(X) 


X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True )

test_data = Data(np.array(X_val), np.array(y_val))
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

print(len(test_data))
print((y_val))

element_count = Counter(y_val)

# print(element_count)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim_1)
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.layer_3 = nn.Linear(hidden_dim_2,output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.layer_3(x)

        return x
       
model = Net(input_dim, hidden_dim_1, hidden_dim_2, output_dim)

learning_rate = 0.01

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


batch_size = 32
batches_per_epoch = len(X) // batch_size
 
best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
    # set model in evaluation mode and run through the test set
    model.eval()
    y_pred = model(X_val)
    ce = loss_fn(y_pred, y_val)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")
    
model.load_state_dict(best_weights)

print("Training Complete")

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_dataloader:
    output = model(inputs) # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
    
    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

# constant for classes
classes = ('none', 'power', 'precision')

# Build confusion matrix
y_true = [np.argmax(i) for i in y_true]
element_count = Counter(y_true)
print(element_count)
print('F1', f1_score(y_true, y_pred, average=None))
print('F1-micro', f1_score(y_true, y_pred, average='micro'))
print('F1-macro', f1_score(y_true, y_pred, average='macro'))
print('F1-weighted', f1_score(y_true, y_pred, average='weighted'))

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot=True, cmap='Blues')
plt.savefig('output.png')

