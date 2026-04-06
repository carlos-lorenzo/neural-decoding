import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate


class SNN(nn.Module):
    def __init__(self, electrode_channels: int, hidden_features: int, n_classes: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(electrode_channels, hidden_features)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())

        self.fc2 = nn.Linear(hidden_features, n_classes)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, mem1: torch.Tensor, mem2: torch.Tensor) -> tuple:
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        return mem1, mem2, spk2

    def reset_mem(self) -> tuple:
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        return mem1, mem2
