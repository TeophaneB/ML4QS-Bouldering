import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_tcn import TCN
import pandas as pd

# =====================================================================
# 1. DEFINE CUSTOM DATASET FOR SEQUENCES
# =====================================================================
class BoulderingSequenceDataset(Dataset):
    """
    Accepts raw sequence data and handles padding and shape formatting 
    specifically for pytorch-tcn ('NCL' structure).
    """
    def __init__(self, list_of_sequences, list_of_labels):
        """
        Args:
            list_of_sequences: List of 2D numpy arrays/tensors, 
                               each of shape (sequence_length, num_channels)
            list_of_labels: List of integer target classes (0 to num_classes-1)
        """
        # 1. Transpose each sequence from (Length, Channels) -> (Channels, Length)
        # because pytorch-tcn expects the Channel dimension to come first ('NCL')
        processed_seqs = [
            torch.tensor(seq, dtype=torch.float32).t() for seq in list_of_sequences
        ]
        
        # 2. Pad sequences automatically with 0.0 so they all match the maximum length
        # batch_first=True makes the final shape: (Num_Examples, Channels, Max_Length)
        self.sequences = pad_sequence(processed_seqs, batch_first=True, padding_value=0.0)
        self.labels = torch.tensor(list_of_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =====================================================================
# 2. INITIALIZE DATALOADERS (Example Mock-up Data Setup)
# =====================================================================

Dirs = { # dimensions respectively: (num_samples, sequence_length, num_channels) = (43, 500, 15) (11, 500, 15) (43,) (11,)
    "X_train": "deepLearningCode\\X_train_scaled.csv",
    "y_train": "deepLearningCode\\y_train.csv",
    "X_test": "deepLearningCode\\X_test_scaled.csv",
    "y_test": "deepLearningCode\\y_test.csv"
}


import pandas as pd
import torch
import numpy as np

# 1. Load the flat 2D data from CSVs
# (Drop any index columns if your CSVs have an unnamed first column)
X_train_raw = pd.read_csv(Dirs["X_train"], header=None).values
y_train_raw = pd.read_csv(Dirs["y_train"], header=None).values.flatten()

X_test_raw = pd.read_csv(Dirs["X_test"], header=None).values
y_test_raw = pd.read_csv(Dirs["y_test"], header=None).values.flatten()

# 2. Reshape from 2D flat rows to 3D temporal arrays
# Shape changes from (43, 7500) -> (43, 500, 15)
X_train_3d = X_train_raw.reshape(-1, 500, 15)
X_test_3d = X_test_raw.reshape(-1, 500, 15)

print("Reshaped X_train shape:", X_train_3d.shape) # Should output: (43, 500, 15)
print("Reshaped X_test shape:", X_test_3d.shape)   # Should output: (11, 500, 15)

# 3. Convert to a list of 2D arrays to pass to your BoulderingSequenceDataset
# This splits the 3D tensor along the batch dimension into a list of individual trials
train_sequences = [sample for sample in X_train_3d]
test_sequences = [sample for sample in X_test_3d]

# 4. Pass to your dataset
# Your dataset class already has .t() built inside it, which converts each 
# individual sequence from (500, 15) -> (15, 500). 
train_dataset = BoulderingSequenceDataset(train_sequences, y_train_raw)
test_dataset = BoulderingSequenceDataset(test_sequences, y_test_raw)

# Build the PyTorch DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=4,       # Adjust based on your GPU/CPU hardware memory
    shuffle=True,       # Shuffle the samples every epoch
    drop_last=False
)


# =====================================================================
# 3. INSTANTIATE YOUR MODEL CONFIGURATION
# =====================================================================
NUM_CLASSES = 3  # Example: Easy, Medium, Hard

model = TCN(
    num_inputs=15,               # Matching your 15 sensor channels
    num_channels=[32, 64, 128],  # Expands feature representation deeper into the network
    kernel_size=4,
    dilations=None,              # Automatically calculated
    dilation_reset=None,
    dropout=0.1,
    causal=True,                 # Prevents looking ahead into future time-steps
    use_norm='weight_norm',
    activation='relu',
    kernel_initializer='xavier_uniform',
    use_skip_connections=True,   # Helps gradients flow nicely in deep networks
    input_shape='NCL',           # Batch (N), Channels (C), Length (L)
    embedding_shapes=None,       # No extra metadata layers
    embedding_mode='add',
    use_gate=False,
    lookahead=0,
    output_projection=None,
    output_activation=None,
)

# Because we are doing classification, we add a simple Linear head to 
# project the final hidden states into our class dimensions.
classification_head = nn.Linear(128, NUM_CLASSES) # 128 matches the last number in num_channels

# Move your network to your target compute device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
classification_head.to(device)


# =====================================================================
# 4. TRAINING LOOP CONFIGURATION
# =====================================================================
criterion = nn.CrossEntropyLoss()
# Optimize parameters for both the TCN backbone and the classification head
optimizer = optim.Adam(
    list(model.parameters()) + list(classification_head.parameters()), 
    lr=0.001
)

EPOCHS = 10

print("Starting TCN Sequence Classification Training...")
for epoch in range(EPOCHS):
    model.train()
    classification_head.train()
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_sequences, batch_labels in train_loader:
        # Transfer tensors to running device
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device)
        
        # Clear any accumulated gradients
        optimizer.zero_grad()
        
        # 1. Forward Pass through TCN Backbone
        # Output shape: (Batch Size, Final Channels, Sequence Length)
        tcn_output = model(batch_sequences)
        
        # 2. Slice the final relevant index
        # For sequence classification, we pull out the representation at the very 
        # last valid time step index (-1) of the sequence.
        final_timestep_features = tcn_output[:, :, -1] 
        
        # 3. Forward Pass through Linear Classification Head
        logits = classification_head(final_timestep_features)
        
        # 4. Calculate Loss & Optimize
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        # Metrics Tracking
        running_loss += loss.item() * batch_sequences.size(0)
        _, predicted_classes = torch.max(logits, 1)
        correct_predictions += (predicted_classes == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    print(f"Epoch [{epoch+1}/{EPOCHS}] -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# =====================================================================
# 5. EVALUATION ON TEST SET (Optional)
# =====================================================================
model.eval()
classification_head.eval()
test_loss = 0.0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch_sequences, batch_labels in DataLoader(test_dataset, batch_size=4):
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device)
        
        tcn_output = model(batch_sequences)
        final_timestep_features = tcn_output[:, :, -1]
        logits = classification_head(final_timestep_features)
        
        loss = criterion(logits, batch_labels)
        test_loss += loss.item() * batch_sequences.size(0)
        
        _, predicted_classes = torch.max(logits, 1)
        correct_predictions += (predicted_classes == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

test_loss /= total_samples
test_acc = (correct_predictions / total_samples) * 100
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
