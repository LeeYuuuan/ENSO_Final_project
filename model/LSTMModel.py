
import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, output_size=24):
        super(LSTMModel, self).__init__()
        # Define the input size of the linear layer
        self.flatten_input_size = 4 * 24 * 72
        # Define the output size of the linear layer
        self.flatten_output_size = 256
        # Linear layer to compress spatial features
        self.flatten_input = nn.Linear(self.flatten_input_size, self.flatten_output_size)
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(input_size=self.flatten_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape the input to combine spatial dimensions (24x72) into a single dimension
        x = x.view(batch_size, 4, 12, -1)  # Shape: [N, 4, 12, 24*72]
        # Rearrange dimensions to place time (12 months) as the second dimension
        x = x.permute(0, 2, 1, 3)  # Shape: [N, 12, 4, 24*72]
        # Flatten feature dimensions (4x24x72) into a single dimension
        x = x.reshape(batch_size, 12, -1)  # Shape: [N, 12, 4*24*72]
        # Apply linear transformation to compress features
        x = torch.relu(self.flatten_input(x))  # Shape: [N, 12, 256]
        # Pass through LSTM layers
        out, _ = self.lstm(x)  # Shape: [N, 12, hidden_size]
        # Extract the output of the last time step
        out = out[:, -1, :]  # Shape: [N, hidden_size]
        # Map to the desired output size
        out = self.fc(out)  # Shape: [N, 24]
        return out