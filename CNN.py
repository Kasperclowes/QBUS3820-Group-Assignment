# Requirements 

import random

import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch 
import time

print(f'Package versions: \n')

print(f'numpy {np.__version__}')
print(f'pandas {pd.__version__}')
print(f'scikit-learn {sklearn.__version__}')
print(f'torch {torch.__version__}')

# Notebook tested on:

# numpy 1.26.4
# pandas 2.2.3
# scikit-learn 1.6.1
# torch 2.6.0

from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.utils.data import TensorDataset



class NeuralNetwork(nn.Module):
    def __init__(self, input_features=31, hidden_size=128):
        super(NeuralNetwork, self).__init__()

        # Define the network architecture
        self.feedforward = nn.Sequential(            
            nn.Linear(input_features, hidden_size),  # Input layer: features to hidden neurons           
            nn.ReLU(),                               # ReLU activation              
            nn.Linear(hidden_size, hidden_size),     # Hidden layer: hidden to hidden neurons
            nn.ReLU(),                               # ReLU activation
            nn.Linear(hidden_size, hidden_size),     # Hidden layer: hidden to hidden neurons
            nn.ReLU(),                               # ReLU activation
            nn.Linear(hidden_size, 1),               # Output layer: hidden to 1 neuron 
            nn.Sigmoid()                             # Sigmoid activation to get probability
        )                        

    def forward(self, x):        
        # Define the forward pass
        return self.feedforward(x).flatten()  # Flatten to get a 1D tensor
    
    def predict_proba(self, X):
        """
        Get probability predictions for both classes.
        Returns array of shape (n_samples, 2) with [prob_class_0, prob_class_1]
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Convert to tensor if needed
        if isinstance(X, pd.DataFrame):
            X = torch.from_numpy(X.values).float()
        elif isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(device)
        
        self.eval()
        with torch.no_grad():
            probs_class_1 = self(X).cpu().numpy()
        
        # Return probabilities for both classes
        probs_class_0 = 1 - probs_class_1
        return np.column_stack([probs_class_0, probs_class_1])
    
    def predict(self, X, threshold=0.5):
        """
        Get binary predictions using the specified threshold.
        Returns array of 0s and 1s
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


def choose_model_features(X_train):
    num_features = X_train.shape[1]

# Instantiate the model with custom parameters
    model = NeuralNetwork(input_features=num_features, hidden_size=128)

    return model

def create_dataloaders(X_train, y_train, X_valid, y_valid): 

    train_dataset  = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    valid_dataset  = TensorDataset(torch.from_numpy(X_valid.values).float(), torch.from_numpy(y_valid.values).float())
   
# Create a DataLoader for training data
    batch_size = 1024
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)
    
    

    # Check the DataLoader
    for batch_idx, (features, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        # Only print the first batch
    
    return train_loader, valid_dataset

from IPython.display import clear_output, display, HTML

from torch.nn import functional as F
from sklearn.metrics import recall_score, average_precision_score

def evaluate(model, dataset, y_valid, threshold = 0.1):
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set model to evaluation mode (not necessary here but required in general)
    model.eval()
    
    # Get data
    X, y = dataset[:]
    X, y = X.to(device), y.numpy()
        
    # Make predictions (no gradient calculation needed)
    with torch.no_grad():
        y_prob = model(X).cpu().numpy()
        
    # Calculate loss
    loss = F.binary_cross_entropy(
        torch.tensor(y_prob.astype(float)), 
        torch.tensor(y.astype(float))
    ).item()
     
    # Classification using the decision threshold
    y_pred = (y_prob > threshold).astype(int)
    
    # Validation metrics
    recall = recall_score(y_valid, y_pred)
    avg_precision = average_precision_score(y_valid, y_prob)
    
    return loss, recall, avg_precision 

def init_training_table(num_epochs):
    table = pd.DataFrame(np.arange(1, num_epochs+1), columns = ['epoch'])
    table['train loss'] = 0.0
    table['valid loss'] = 0.0
    table['valid recall'] = 0.0
    table['valid average precision'] = 0.0
    table['time'] = ''
    return table

def update_training_table(table, net, validset, epoch, duration, y_valid):
    
    # Run evaluation function to get validation metrics
    valid_loss, valid_recall, valid_avg_precision = evaluate(net, validset, y_valid)
        
    # Update table
    table.iloc[epoch, 2] = np.round(10*valid_loss, 3)
    table.iloc[epoch, 3] = np.round(valid_recall, 3)
    table.iloc[epoch, 4] = np.round(valid_avg_precision, 3)
     
    # Epoch length   
    if duration > 3600:
        table.iloc[epoch, 5] = time.strftime('%H:%M:%S', time.gmtime(duration))
    else:
        table.iloc[epoch, 5] = time.strftime('%M:%S', time.gmtime(duration))
        
    clear_output(wait=True)
    display(HTML(table.iloc[:epoch+1, :].to_html(index=False)))
    
    return table




def train(model, train_loader, valid_dataset, y_valid, num_epochs = 5 , lr = 1e-3):
    
    import random
    import numpy as np
    import torch

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Instantiate model and move to device
    model = model.to(device)
    
    # Loss function
    loss_fn = nn.BCELoss() # binary cross-entropy loss, assumes that the output of the network is a probability
    
    # Instantiate optimiser
    # Adam is a variant of SGD that often works well for training neural networks
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
    
    # Addding a learning rate scheduler to improve training
    # Adam + OneCycleLR is a good default for many problems
    # Learn more: https://sgugger.github.io/the-1cycle-policy.html
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = lr, 
                                                   steps_per_epoch=len(train_loader), epochs = num_epochs,
                                                   three_phase=True)
    # Number of training samples
    num_samples = len(train_loader.dataset)
    
    # Initialise table to track training
    table =  init_training_table(num_epochs)
    
    # Training loop
    print('Running first epoch')
    for epoch in range(num_epochs):
        
        # Train phase
        model.train()
        train_loss = 0.0
        
        # Initialise timer
        epoch_start = time.time()    
        
        # Iterate over minibatches
        for X_batch, y_batch in train_loader:

            # Move minibatch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # Calculate loss
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Calculate gradients
            optimizer.step()       # Update weights

            # Update scheduler
            scheduler.step()

            # Accumulate batch loss
            train_loss += loss.item() * len(y_batch)
            

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
  
        # Epoch length
        duration = time.time() - epoch_start 
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Duration: {duration:.2f} seconds')
        
        # Display metrics
        table.iloc[epoch, 1] = np.round(10*train_loss, 3)
        table =  update_training_table(table, model, valid_dataset, epoch, duration, y_valid)
    
    return model



def predict(model, X):
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Move data to device
    X_g = X.to(device)
    
    # Put model on evaluation mode (it makes no difference but needed in some cases)
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():
        
        # Predicted probabilities 
        # the .cpu().detach() part transfers the result to the cpu
        output  = model(X_g).cpu().detach()
    
    return output # the output is a tensor