import torch
import torch.nn as nn


class EEGNetBlock(nn.Module):
    """
    This module abstracts a branch of the Fusion EEGNet model.
    It follows the architecture of the original EEGNet, but with some modifications to fit the fusion model.

    Args:
    - electrode_channels (int): The number of EEG channels (electrodes) in the input data.
    - temporal_filters (int): The number of filters in the temporal convolution layer.
    - spatial_filters (int): The number of filters in the spatial convolution layer.
    - temporal_kernel_length (int): The length of the temporal convolution kernel.
    - out_channels (int): The number of output channels after the pointwise convolution, which determines the feature dimension for the classification head.
    - dropout (float, optional): The dropout rate applied after the second pooling layer to prevent overfitting. Default is 0.5.
    """

    def __init__(
        self,
        electrode_channels: int,
        temporal_filters: int,
        spatial_filters: int,
        temporal_kernel_length: int,
        out_channels: int,
        dropout: float = 0.5
    ) -> None:
        super(EEGNetBlock, self).__init__()

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=temporal_filters,
            kernel_size=(1, temporal_kernel_length),
            padding='same',
            bias=False
        )
        self.batch_norm_temporal = nn.BatchNorm2d(temporal_filters)

        self.spatial_conv = nn.Conv2d(
            in_channels=temporal_filters,
            out_channels=spatial_filters,
            kernel_size=(electrode_channels, 1),
            groups=temporal_filters,
            bias=False
        )

        self.batch_norm_spatial = nn.BatchNorm2d(spatial_filters)

        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, temporal_kernel_length * 2),
                padding='same',
                groups=spatial_filters,
                bias=False
            ),
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            )
        )

        self.batch_norm_separable = nn.BatchNorm2d(out_channels)

        self.pool2 = nn.AvgPool2d((1, 8))

        self.dropout = nn.Dropout(p=dropout)

    def output_features(self, sample_length: int) -> int:
        """
        Returns the integer number of output features produced by this block
        for a given `sample_length` (temporal length).

        Because the pooling layers use kernel sizes of 4 then 8 (with default
        stride equal to kernel size), the temporal dimension after pooling is
        reduced by a factor of 4 and then 8. When `sample_length` is not
        perfectly divisible by these factors we use floor division to compute
        the resulting temporal size.

        The final number of features equals `out_channels * floor(sample_length / 32)`.
        This method returns that integer value (suitable for feeding into a
        classifier head or for flatten size calculations).
        """

        reduction = 4 * 8
        temporal_after_pool = sample_length // reduction

        return int(self.separable_conv[-1].out_channels * temporal_after_pool)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.batch_norm_temporal(x)
        x = self.spatial_conv(x)
        x = self.batch_norm_spatial(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.separable_conv(x)
        x = self.batch_norm_separable(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        return x


class FusionEEGNet(nn.Module):
    """
    This module implements the Fusion EEGNet architecture, which consists of multiple parallel branches (EEGNetBlocks) that process data in parallel with different filter configurations.
    """

    def __init__(self, n_classes: int, electrode_channels: int, sample_length: int, filters: list[tuple[int, int, int, int]]):
        """

        Args:
            n_classes (int): The number of output classes for classification.
            electrode_channels (int): The number of EEG channels (electrodes) in the input data.
            sample_length (int): The temporal length of the input data (number of time points).
            filters (list[tuple[int, int, int, int]]): A list where each tuple contains the configuration for a branch in the format (temporal_filters, spatial_filters, temporal_kernel_length, out_channels).
        """
        super(FusionEEGNet, self).__init__()
        self.branches = nn.ModuleList([
            EEGNetBlock(
                electrode_channels=electrode_channels,
                temporal_filters=temporal_filters,
                spatial_filters=spatial_filters,
                temporal_kernel_length=temporal_kernel_length,
                out_channels=out_channels
            )
            for temporal_filters, spatial_filters, temporal_kernel_length, out_channels in filters
        ])

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            in_features=sum(branch.output_features(sample_length)  # type: ignore
                            for branch in self.branches),
            out_features=n_classes
        )

    def forward(self, x):
        x = [branch(x) for branch in self.branches]
        x = torch.cat(x, dim=1)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
