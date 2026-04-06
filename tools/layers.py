import torch
import torch.nn as nn
import numpy as np


class STRS(nn.Module):
    """
        Sparsity-Targeted Robust Scaling (STRS) is a novel normalization technique designed to enhance the robustness of EEGNet models when applied to EEG data. STRS dynamically adjusts its scaling parameters based on the distribution of the input data, making it particularly effective for handling the variability and noise commonly found in EEG signals.

        Args:
            n_classes (int): The number of classes in the classification task. This is used to calculate the gamma parameter, which controls the scaling based on the expected sparsity of the data.
            momentum (float, optional): Momentum acts as a "smoothing filter" for your data's statistical profile. It controls how much the model trusts the current batch of EEG data versus the historical average of all data it has seen so far. Defaults to 0.1.
    """

    def __init__(self, n_classes: int, momentum: float = 0.1):

        super().__init__()
        self.n_classes = n_classes
        self.momentum = momentum

        # gamma is static
        self.gamma = float(np.log(n_classes - 1))

        # Learnable/running stats
        self.register_buffer('running_median', torch.tensor(0.0))
        self.register_buffer('running_q3', torch.tensor(1e-6))
        self.register_buffer('running_kappa', torch.tensor(1.0))
        self.first_pass = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = torch.abs(x)

        if self.training:
            # Calculate current batch stats
            # Flatten to calculate global median across the batch features
            batch_median = torch.median(x_abs)
            x_centered = x_abs - batch_median
            batch_q3 = torch.quantile(x_centered, 0.75)

            if batch_q3 <= 0:
                batch_q3 = torch.tensor(1e-6, device=x.device)

            batch_kappa = (self.gamma + np.log(3)) / batch_q3

            # Update running stats
            if self.first_pass:
                self.running_median.copy_(batch_median)
                self.running_q3.copy_(batch_q3)
                self.running_kappa.copy_(batch_kappa)
                self.first_pass = False
            else:
                self.running_median.copy_(
                    (1 - self.momentum) * self.running_median + self.momentum * batch_median)
                self.running_q3.copy_(
                    (1 - self.momentum) * self.running_q3 + self.momentum * batch_q3)
                self.running_kappa.copy_(
                    (1 - self.momentum) * self.running_kappa + self.momentum * batch_kappa)

            median = batch_median
            kappa = batch_kappa
        else:
            median = self.running_median
            kappa = self.running_kappa

        x_centered = x_abs - median
        probabilities = 1 / (1 + torch.exp(-(kappa * x_centered - self.gamma)))

        return probabilities
