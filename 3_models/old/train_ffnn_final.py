import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score,plot_precision_recall_curve, plot_roc_curve
import argparse
import os
import pdb
import itertools
import json

from torch_classes import FeatureDataset, FFNeuralNet


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="", type=str, help='directory with sparce matrices train val test')
parser.add_argument('--label', default="label_max24", type=str, help='the column name for label in traige_TE.triage_cohort_final_with_labels_complete1vs')
parser.add_argument('--model_file', default = "", type=str, help = "JSON file of validated model hyperparameters")
args = parser.parse_args()


with open(os.path.join(args.model_file, 'best_params.json')) as f:
  best_params = json.load(f)

lr = best_params['learning_rate']
l2 = best_params['lambda']
num_epochs = best_params['num_epochs']
drop_out = best_params['drop_out']

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True}
max_epochs = 100

#Datasets
train_set = FeatureDataset(os.path.join(args.data_dir, 'training_and_val_examples.npz'),
                           os.path.join(args.data_dir, 'train_and_val_labels.csv'),
                           args.label)
test_set = FeatureDataset(os.path.join(args.data_dir, 'test_examples.npz'),
                          os.path.join(args.data_dir, 'test_labels.csv'),
                          args.label)

# DataLoaders
train_loader = DataLoader(train_set, **params)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# Define Model
model = FFNeuralNet(train_set.input_size, hidden_size=1024, drop_out=drop_out).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

# Train the model
total_step = len(train_loader)
model.train()
for epoch in range(num_epochs):

    for i, (features, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = criterion(outputs, torch.reshape(labels, (labels.shape[0], 1)))
        loss += l2 * l2_reg

        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Evaluate the Model
model.eval()

all_outputs = []
all_labels = []
model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for features, labels in test_loader:
	    features = features.to(device)
	    labels = labels.to(device)
	    outputs = model(features)
	    all_outputs += [o[0] for o in outputs.detach().cpu().numpy()]
	    all_labels += [l for l in labels.detach().cpu().numpy()]

auroc = roc_auc_score(all_labels, all_outputs)
print("AUROC %.2f" % auroc)


df_labels = pd.read_csv(os.path.join(args.data_dir,'test_labels.csv'))
df_labels = (df_labels
    .assign(label=all_labels,
            predictions=all_outputs)
)

os.makedirs(os.path.join(args.model_file, 'test'), exist_ok=True)
df_labels.to_csv(os.path.join(args.model_file, 'test', 'ffnn_results.csv'))

