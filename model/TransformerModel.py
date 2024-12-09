

import torch
from torch import nn

class TransformerModel(nn.Module):
    def __init__(self, spatial_dim=24*72, embed_dim=256, num_heads=8, num_layers=4, output_dim=24):
        super(TransformerModel, self).__init__()
        # Spatial features are compressed into embed_dim
        self.flatten_input = nn.Linear(4 * spatial_dim, embed_dim)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final prediction layer
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # Flatten spatial dimensions (24x72) into one dimension
        x = x.view(batch_size, 4, 12, -1)  # Shape: [N, 4, 12, 24*72]
        # Rearrange dimensions to combine feature channels into spatial features
        x = x.permute(0, 2, 1, 3)  # Shape: [N, 12, 4, 24*72]
        # Flatten feature dimensions (4x24*72) into a single dimension
        x = x.reshape(batch_size, 12, -1)  # Shape: [N, 12, 4*24*72]
        # Linear transformation to compress features
        x = self.flatten_input(x)  # Shape: [N, 12, embed_dim]
        # Pass through the Transformer encoder
        x = self.transformer(x)  # Shape: [N, 12, embed_dim]
        # Use the last time step's output for prediction
        x = x[:, -1, :]  # Shape: [N, embed_dim]
        # Map to the desired output size
        x = self.fc(x)  # Shape: [N, 24]
        return x


class TransformerModel_CNN(nn.Module):
    def __init__(self, spatial_dim=24*72, embed_dim=256, num_heads=8, num_layers=4, output_dim=24):
        super(TransformerModel_CNN, self).__init__()
        # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv3d(4, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)

        # Flatten spatial features into a vector
        self.flatten_input = nn.Linear(64 * spatial_dim, embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final prediction layer
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)  # Shape: [N, 64, 12, 24, 72]

        # Flatten spatial dimensions for transformer input
        x = x.view(batch_size, 64, 12, -1)  # Shape: [N, 64, 12, 24*72]
        x = x.permute(0, 2, 1, 3)  # Shape: [N, 12, 64, 24*72]
        x = x.reshape(batch_size, 12, -1)  # Shape: [N, 12, 64*24*72]

        # Linear transformation to compress features
        x = self.flatten_input(x)  # Shape: [N, 12, embed_dim]

        # Pass through the Transformer encoder
        x = self.transformer(x)  # Shape: [N, 12, embed_dim]

        # Use the last time step's output for prediction
        x = x[:, -1, :]  # Shape: [N, embed_dim]

        # Map to the desired output size
        x = self.fc(x)  # Shape: [N, 24]
        return x