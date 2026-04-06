import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, GATv2Conv
import copy


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_kernel_size=9, K=3):
        super(STGCNBlock, self).__init__()

        # Spatial convolution using Chebyshev Polynomials
        self.spatial_conv = ChebConv(in_channels, out_channels, K=K)

        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(1, time_kernel_size),
            padding='same'
        )

        # Batch normalization and activation
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: (Batch Size, Channels, Nodes, Time Samples)

        # 1. Spatial Convolution (GNN processing)
        # Reshape for PyG: (Batch Size * Time Samples, Channels, Nodes)
        batch_size, channels, nodes, time_samples = x.shape

        # We first flatten the batch and time dimensions into the total nodes across disconnected graphs:
        x_spatial = x.permute(0, 3, 2, 1).reshape(
            batch_size * time_samples * nodes, channels)

        # Create a batched edge_index for (batch_size * time_samples) graphs
        num_graphs = batch_size * time_samples
        device = x.device
        offsets = torch.arange(num_graphs, device=device) * nodes
        edge_index_batched = edge_index.repeat(
            1, num_graphs) + offsets.repeat_interleave(edge_index.size(1))

        if edge_weight is not None:
            edge_weight_batched = edge_weight.repeat(num_graphs)
        else:
            edge_weight_batched = None

        # Apply ChebConv
        spatial_out = self.spatial_conv(
            x_spatial, edge_index_batched, edge_weight_batched)

        # Reshape back to (Batch, Channels, Nodes, Time)
        spatial_out = spatial_out.reshape(
            batch_size, time_samples, nodes, -1).permute(0, 3, 2, 1)

        # 2. Temporal Convolution
        # Temporal conv operates on (Batch, 1, Nodes, Time) or similar, depending on treating nodes as channels or spatial dims.
        # Original STGCN treats input as (Batch, In_Channels, Nodes, Time).
        # To match the Conv2d, we can use the original x.
        # But wait, typically it's parallel or sequential. Let's do sequential.
        # But wait, original STGCN often uses Temporal -> Spatial -> Temporal.
        # For simplicity, let's treat the out_channels of spatial as input to temporal.

        # Reshape spatial_out to (Batch, out_channels, Nodes, Time) to act as standard 2D feature map
        # Wait, temporal conv takes in_channels=1 if we process each out_channel independently, or in_channels=out_channels.
        # Let's fix temporal_conv to take `out_channels` from spatial conv.

        temporal_out = self.temporal_conv(spatial_out)

        # 3. Residual connection and activation
        out = self.batch_norm(temporal_out)
        out = self.relu(out)

        return out


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, pooling=(1, 4), dropout_rate=0.5):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(1, kernel_size), padding='same')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(pooling)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.dropout(self.pool(self.elu(self.batch_norm(self.conv(x)))))


class STGCN(nn.Module):
    def __init__(self, n_channels, n_classes, time_samples, K=3, dropout_rate=0.5):
        super(STGCN, self).__init__()

        # Initial Temporal Convolution
        self.tcn1 = TemporalConvLayer(
            in_channels=1, out_channels=16, kernel_size=64, pooling=(1, 4), dropout_rate=dropout_rate)

        # STGCN Blocks
        self.spatial_conv = ChebConv(16, 16, K=K)
        self.spatial_bn = nn.BatchNorm2d(16)
        self.spatial_elu = nn.ELU()
        self.spatial_dropout = nn.Dropout(p=dropout_rate)

        # Second Temporal Convolution
        self.tcn2 = TemporalConvLayer(
            in_channels=16, out_channels=32, kernel_size=16, pooling=(1, 8), dropout_rate=dropout_rate)

        self.flatten = nn.Flatten()

        # Calculate resulting temporal dimension
        reduced_time = time_samples // 32
        feature_dim = 32 * n_channels * reduced_time

        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x, edge_index, edge_weight=None):
        # x shape initially from dataloader: (Batch, 1, Channels (Nodes), Time)
        batch_size, _, nodes, time_samples = x.shape

        # 1. Initial Temporal Convolution
        x = self.tcn1(x)  # Shape: (Batch, 16, Nodes, Time//4)

        # Update current time dimension length after pooling
        _, channels, _, current_time = x.shape

        # 2. Spatial Convolution
        # Flatten node features: (Batch * Time * Nodes, Channels)
        x_spatial = x.permute(0, 3, 2, 1).reshape(
            batch_size * current_time * nodes, channels)

        num_graphs = batch_size * current_time
        device = x.device
        offsets = torch.arange(num_graphs, device=device) * nodes
        edge_index_batched = edge_index.repeat(
            1, num_graphs) + offsets.repeat_interleave(edge_index.size(1))

        if edge_weight is not None:
            edge_weight_batched = edge_weight.repeat(num_graphs)
        else:
            edge_weight_batched = None

        spatial_out = self.spatial_conv(
            x_spatial, edge_index_batched, edge_weight_batched)

        # Reshape back to (Batch, 16, Nodes, Time)
        x = spatial_out.reshape(batch_size, current_time,
                                nodes, channels).permute(0, 3, 2, 1)

        # Apply activation, batch norm, and dropout for spatial convolution block
        x = self.spatial_dropout(self.spatial_elu(self.spatial_bn(x)))

        # 3. Second Temporal Convolution
        x = self.tcn2(x)  # Shape: (Batch, 32, Nodes, Time//32)

        # 4. Pooling and Classification
        x = self.flatten(x)  # Shape: (Batch, Feature_Dim)
        out = self.fc(x)

        return out
