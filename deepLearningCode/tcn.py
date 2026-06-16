import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    Slices off the trailing padding elements of a 1D convolution output.
    This ensures the convolution is strictly causal—preventing the model 
    from looking into the "future" of the time series sequence.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Tensor shape input:  (Batch, Channels, Length + Padding)
        # Tensor shape output: (Batch, Channels, Length)
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single residual block consisting of two dilated causal 1D convolutional layers,
    incorporating weight normalization, ReLU activations, and dropout.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # --- First Convolutional Layer ---
        # Weight norm stabilizes training. Padding is calculated to accommodate the dilation factor.
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # Chomp removes the padded elements from the end to maintain causality
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # --- Second Convolutional Layer ---
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine the sequential operations of the block
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # --- Residual Skip Connection ---
        # If input channels don't match output channels, use a 1x1 convolution to match dimensions
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initializes weights with a normal distribution for stable variance."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # Compute the residual path (identity mapping or 1x1 downsample projection)
        res = x if self.downsample is None else self.downsample(x)
        # Element-wise addition of features and residual before the final activation
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    The main backbone network composed of stacked TemporalBlocks.
    Dilation scales exponentially (1, 2, 4, 8...) at each level to maximize the receptive field.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Stack blocks dynamically based on the length of the num_channels list
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, etc.
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # The causal padding required equals: (kernel_size - 1) * dilation
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Expects input shape: (Batch Size, Features/Channels, Sequence Length)
        return self.network(x)
    

class TCN(nn.Module):
    """
    Sequence Classification Head Wrapper.
    Passes data through the TCN network and maps the final step output to target classes.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # Initialize the feature extraction core
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # Fully connected layer to map final features to output classes (e.g., bouldering difficulty)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        # inputs shape: (Batch, Channels, Length)
        y1 = self.tcn(inputs)  
        
        # y1[:, :, -1] isolates the vector at the very last step index (-1) of the sequence.
        # This represents the accumulated temporal representation of the full bouldering attempt.
        o = self.linear(y1[:, :, -1])
        
        # Returns log probabilities for multi-class classification
        import torch.nn.functional as F
        return F.log_softmax(o, dim=1)
    

"""
import torch.nn.functional as F
from torch import nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        # Inputs have to have dimension (N, C_in, L_in)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)
"""
