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

normalize = True
n_epochs = 300

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

df = pd.read_csv('data/depths_addedZ_final.csv')

####################### df = df.drop('picture_name', axis=1)

# le = preprocessing.LabelEncoder()
# le.fit(['none', 'power', 'precision'])

max_dist = df['depth_dist'].abs().max()
print(df)
def apply_func(row):
    for i in range(0, 8):
        if i % 2 == 0:
            row[i] = row[i] / 1920
        else:
            row[i] = row[i] / 1080
    
    for i in range(8, 71):
        if i % 3 == 2:
            row[i] = row[i] / 1920
        elif i % 3 == 0:
            row[i] = row[i] / 1080

    row[71] = row[71] / max_dist
    
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
# print(df.handedness.value_counts())
# print(df.columns)

X = df.iloc[:, 0:-2]
y = df.iloc[:, -1:]
# y = np.array(df).reshape(-1, 1)
X = pd.get_dummies(X, columns=['handedness'])
print(X)
print(y.value_counts())


print(y)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# print(ohe.categories_)

print(y)
y = ohe.transform(y)
print(ohe.categories_)

y_df = pd.DataFrame(y)
y_df['picture_name'] = df['picture_name']


print(X)
# 1/0
# train_data = Data(np.array(X_train), np.array(y_train))
# train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


input_dim = 74
hidden_dim_1 = 64
hidden_dim_2 = 32
output_dim = 3
# batches_per_epoch = len(X) 

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', y)
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


X_train, X_val, y_train, y_val = train_test_split(X, y_df, test_size=0.15, random_state=42,shuffle=True )

y_val_names = y_val['picture_name']
y_train.drop('picture_name', axis=1, inplace=True)
y_val.drop('picture_name', axis=1, inplace=True)
# y_train = y_train.reset_index()
print(y_train.values.tolist())

y_train = torch.tensor(y_train.values.tolist(), dtype=torch.float32)
y_val = torch.tensor(y_val.values.tolist(), dtype=torch.float32)

test_data = Data(np.array(X_val), np.array(y_val))
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

print(len(test_data))
print(y_val)

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
            # print('\n\n\n')
            # print(y_pred)
            # print('XD')
            # print(y_batch)
            # print('\n\n\n')
            # 1/0
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
    ce = loss_fn(y_pred[-1], y_val[-1])
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
y_name = []
# iterate over test data
i = 0
for inputs, labels in test_dataloader:
    output = model(inputs) # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
    
    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save Truth


# constant for classes
classes = ('none', 'power', 'precision')

# # Build confusion matrix
y_true = [np.argmax(i) for i in y_true]
# print(y_pred)
# print(y_true)
y_val_names = pd.DataFrame(y_val_names)
y_val_names['Prediction'] = y_pred
y_val_names['True'] = y_true

y_val_names.loc[y_val_names['Prediction'] == 0, 'Prediction'] = 'None'
y_val_names.loc[y_val_names['Prediction'] == 1, 'Prediction'] = 'Power'
y_val_names.loc[y_val_names['Prediction'] == 2, 'Prediction'] = 'Precision'

y_val_names.loc[y_val_names['True'] == 0, 'True'] = 'None'
y_val_names.loc[y_val_names['True'] == 1, 'True'] = 'Power'
y_val_names.loc[y_val_names['True'] == 2, 'True'] = 'Precision'

print(y_val_names)


counts = y_val_names['picture_name'].value_counts()
# sort the counts in descending order
y_val_names['counts'] = y_val_names['picture_name'].apply(lambda x: counts[x])
# sort the DataFrame by the counts column and then by the original index
df_sorted = y_val_names.sort_values(['counts', 'picture_name'],ascending=False)#, y_val_names.index])
print(df_sorted)

# drop the counts column if not needed
df_sorted = df_sorted.drop('counts', axis=1)

# count_series = counts.values
# value_series = counts.index

# print(count_series)
# print(value_series)
df_sorted['Equal'] = np.where(df_sorted['Prediction'] == df_sorted['True'], ' ', 'X')
print(df_sorted)

df_sorted.to_csv('validation_results_sorted.csv', index = False)


element_count = Counter(y_true)
print('none', 'power', 'precision')
print(element_count)
print()
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

# ig = IntegratedGradients(model)
# ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
# gs = GradientShap(model)
# fa = FeatureAblation(model)
# print(X_val)
# # X_val = torch.tensor(X_val.values).float()

# ig_attr_test = ig.attribute(X_val, n_steps=5)
# ig_nt_attr_test = ig_nt.attribute(X_val)
# dl_attr_test = dl.attribute(X_val)
# gs_attr_test = gs.attribute(X_val, torch.tensor(X_train.values).float())
# fa_attr_test = fa.attribute(X_val)


# # prepare attributions for visualization

# feature_names = ["l_hand_min_x", "l_hand_min_y", "l_hand_width", "l_hand_height",    "r_hand_min_x", "r_hand_min_y", "r_hand_width", "r_hand_height",    "object_min_x", "object_min_y", "object_width", "object_height",    "l1x", "l1y", "l1z", "l2x", "l2y", "l2z", "l3x", "l3y", "l3z",    "l4x", "l4y", "l4z", "l5x", "l5y", "l5z", "l6x", "l6y", "l6z",    "l7x", "l7y", "l7z", "l8x", "l8y", "l8z", "l9x", "l9y", "l9z",    "l10x", "l10y", "l10z", "l11x", "l11y", "l11z", "l12x", "l12y", "l12z",    "l13x", "l13y", "l13z", "l14x", "l14y", "l14z", "l15x", "l15y", "l15z",    "l16x", "l16y", "l16z", "l17x", "l17y", "l17z", "l18x", "l18y", "l18z",    "l19x", "l19y", "l19z", "l20x", "l20y", "l20z", "l21x", "l21y", "l21z",    "r1x", "r1y", "r1z", "r2x", "r2y", "r2z", "r3x", "r3y", "r3z",    "r4x", "r4y", "r4z", "r5x", "r5y", "r5z", "r6x", "r6y", "r6z",    "r7x", "r7y", "r7z", "r8x", "r8y", "r8z", "r9x", "r9y", "r9z",    "r10x", "r10y", "r10z", "r11x", "r11y", "r11z", "r12x", "r12y", "r12z",    "r13x", "r13y", "r13z", "r14x", "r14y", "r14z", "r15x", "r15y", "r15z",    "r16x", "r16y", "r16z", "r17x", "r17y", "r17z", "r18x", "r18y", "r18z",    "r19x", "r19y", "r19z", "r20x", "r20y", "r20z", "r21x", "r21y", "r21z",    "picture_name", "grasp"]


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