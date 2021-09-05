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

INPUT_SIZE = 32345
LEARNING_RATES= [0.001, 0.0005, 0.0001, 0.00005]
LAMBDA=[1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
DROPOUT=[0.1, 0.2, 0.3, 0.4, 0.5]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="", type=str, help='directory with sparce matrices train val test')
parser.add_argument('--label', default="label_max24", type=str, help='the column name for label in traige_TE.triage_cohort_final_with_labels_complete1vs')
parser.add_argument('--output_dir', default="", type=str, help='directory to save outputs')
args = parser.parse_args()


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True}
max_epochs = 100

#Datasets
train_set = FeatureDataset(os.path.join(args.data_dir, 'training_examples.npz'),
                           os.path.join(args.data_dir, 'training_labels.csv'),
                           args.label)
test_set = FeatureDataset(os.path.join(args.data_dir, 'validation_examples.npz'),
                          os.path.join(args.data_dir, 'validation_labels.csv'),
                          args.label)

# DataLoaders
train_loader = DataLoader(train_set, **params)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
best_params = None
best_auroc = 0.0
best_model_dir = None
max_epochs = 0

for lr, drop_out, l2 in list(itertools.product(LEARNING_RATES, DROPOUT, LAMBDA)):
    print("Training model - learning rate: {}, drop_out: {}, lambda: {}".format(
            lr, drop_out, l2))

    # Make output dir for this model 
    model_dir = os.path.join(args.output_dir,
                             'grid_search',
                             'lr_'+str(lr),
                             'drop_out_'+str(drop_out),
                             'lambda_'+ str(l2))
    os.makedirs(model_dir, exist_ok=True)

    # Define Model
    model = FFNeuralNet(INPUT_SIZE, hidden_size=1024, drop_out=drop_out).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    # Train the model
    total_step = len(train_loader)
    num_epochs = 100 # max number of epochs 
    old_val_loss = 999999999999 # some large number for early stopping
    model.train()
    for epoch in range(num_epochs):

        model.train()
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

        # Save model checkpoint
        torch.save({
        'model_state_dict': model.state_dict(),
        }, os.path.join(model_dir, 'checkpoint_{}'.format(epoch+1)))


        # Test early stopping
        if (epoch+1) % 1 == 0: 
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    val_loss += criterion(outputs, torch.reshape(labels, (labels.shape[0], 1))) # no reg

            print ('Epoch [{}/{}], Val Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, val_loss.item()))

            if val_loss > old_val_loss:
                print("Early  stopping at epoch {}".format(epoch+1))
                print("Loading model from epoch {}".format(epoch))
                model = FFNeuralNet(INPUT_SIZE, hidden_size=1024, drop_out=0.2).to(device)
                checkpoint = torch.load(os.path.join(model_dir, 'checkpoint_{}'.format(epoch)))
                model.load_state_dict(checkpoint['model_state_dict'])
                max_epochs = epoch # store as hyperparamter for future training
                break
        
            old_val_loss = val_loss

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

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

    if best_params == None or auroc > best_auroc:
        best_params = {'learning_rate' : lr,
                       'drop_out' : drop_out,
                       'lambda' : l2,
                       'num_epochs' : max_epochs}
        best_auroc = auroc
        best_model_dir = model_dir

# Load Best Model and Checkpoint, evaluate on val set and save csv. 
model = FFNeuralNet(INPUT_SIZE, hidden_size=1024, drop_out=0.2).to(device)
checkpoint = torch.load(os.path.join(best_model_dir, 'checkpoint_%d' % best_params['num_epochs']))
model.load_state_dict(checkpoint['model_state_dict'])
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
print("Best Params - learning rate: {}, drop_out: {}, lambda: {}, num_epochs: {}".format(
            best_params['learning_rate'], best_params['drop_out'], best_params['lambda'],
            best_params['num_epochs']))
print("Best AUROC %.2f" % auroc)

with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as fp:
    json.dump(best_params, fp)


df_labels = pd.read_csv(os.path.join(args.data_dir,'validation_labels.csv'))
df_labels = (df_labels
    .assign(label=all_labels,
            predictions=all_outputs)
)
df_labels.to_csv(os.path.join(args.output_dir, 'ffnn_results.csv'))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model_filedel.ckpt')


# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     for local_batch, local_labels in training_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         [...]

#     # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             [...]
