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


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

df = pd.read_csv('results/data_final.csv')
df = df.drop('col64', axis=1)

# le = preprocessing.LabelEncoder()
# le.fit(['none', 'power', 'precision'])


# df['col65'] = le.transform(df['col65'])


batch_size = 32
# Tips of the fingers positions
#   |        X           |           Y          |            Z          |
l = [0, 4, 8, 12, 16, 20, 21, 25, 29, 33, 37, 41]# 42, 46, 50, 54, 58, 62]

# df['col65'] = le.transform(df['col65'])

X = df.iloc[:, l]
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


input_dim = 12
hidden_dim_1 = 64
hidden_dim_2 = 32
output_dim = 3
# batches_per_epoch = len(X) 


X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True )


test_data = Data(np.array(X_val), np.array(y_val))
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


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


n_epochs = 30
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
# print(y_true)
# print('\n')
# print(y_pred)
y_true = [np.argmax(i) for i in y_true]


cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot=True)
plt.savefig('output.png')










# metric = ConfusionMatrix(num_classes=3)
# metric.attach(default_evaluator, 'cm')
# y_true = torch.tensor([0, 1, 0, 1, 2])
# y_pred = torch.tensor([
#     [0.0, 1.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [0.0, 1.0, 0.0],
# ])
# state = default_evaluator.run([[y_pred, y_true]])
# print(state.metrics['cm'])

# step = np.linspace(0, 100, 1296)







# fig, ax = plt.subplots(figsize=(8,5))
# plt.plot(step, np.array(loss_values))
# plt.title("Step-wise Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()


# y_pred = []
# y_test = []

# total = 0
# correct = 0

# with torch.no_grad():
#     for X, y in test_dataloader:
#         outputs = model(X)
#         predicted = np.where(outputs < 0.5, 0, 1)
#         predicted = list(itertools.chain(*predicted))
#         y_pred.append(predicted)
#         y_test.append(y)
#         total += y.size(0)
#         correct += (predicted == y.numpy())

# print(f'Accuracy of the network on the {len(X_val)} test instances: {100 * correct // total}%')



# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# import seaborn as sns


# y_pred = list(itertools.chain(*y_pred))
# y_test = list(itertools.chain(*y_test))
# print(y_test)
# print(y_pred)

# print(classification_report(y_test, y_pred))


# cf_matrix = confusion_matrix(y_test, y_pred)

# plt.subplots(figsize=(8, 5))

# sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

# plt.show()

# ig = IntegratedGradients(model)
# ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
# gs = GradientShap(model)
# fa = FeatureAblation(model)

# X_val = torch.tensor(X_val.values).float()

# ig_attr_test = ig.attribute(X_val, n_steps=5)
# ig_nt_attr_test = ig_nt.attribute(X_val)
# dl_attr_test = dl.attribute(X_val)
# gs_attr_test = gs.attribute(X_val, torch.tensor(X_train.values).float())
# fa_attr_test = fa.attribute(X_val)


# # prepare attributions for visualization

# feature_names = ['wrist_x', 
#                  'thu_tip_x', 
#                  'ind_tip_x',
#                  'mid_tip_x', 
#                  'ring_tip_x', 
#                  'pinky_tip_x',
#                  'wrist_y', 
#                  'thu_tip_y', 
#                  'ind_tip_y',
#                  'mid_tip_y', 
#                  'ring_tip_y', 
#                  'pinky_tip_y',
#                  'wrist_z', 
#                  'thu_tip_z', 
#                  'ind_tip_z',
#                  'mid_tip_z', 
#                  'ring_tip_z', 
#                  'pinky_tip_z']


# x_axis_data = np.arange(X_val.shape[1])
# x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

# ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
# ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

# ig_nt_attr_test_sum = ig_nt_attr_test.detach().numpy().sum(0)
# ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

# dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
# dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

# gs_attr_test_sum = gs_attr_test.detach().numpy().sum(0)
# gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

# fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
# fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

# lin_weight = model.layer_1.weight[0].detach().numpy()
# y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

# width = 0.14
# legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']

# plt.figure(figsize=(20, 10))

# ax = plt.subplot()
# ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
# ax.set_ylabel('Attributions')

# FONT_SIZE = 16
# plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
# plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
# plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
# plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

# ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
# ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
# ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
# ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
# ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
# ax.autoscale_view()
# plt.tight_layout()

# ax.set_xticks(x_axis_data + 0.5)
# ax.set_xticklabels(x_axis_data_labels)

# plt.legend(legends, loc=3)
# plt.show()
