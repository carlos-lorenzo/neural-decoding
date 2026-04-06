import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from snntorch import spikeplot as snn_spikeplot
except Exception:
    snn_spikeplot = None


def visualize_snn_activity(
    spk_rec: torch.Tensor,
    mem_rec: torch.Tensor,
    sample: int = 0,
    output_neurons: Optional[Iterable[int]] = None,
    figsize: Tuple[int, int] = (14, 9),
    palette: str = "tab10",
    title: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Visualize SNN activity (membrane potential, spikes) for a single sample.

    Args:
        spk_rec: Tensor of shape (time_steps, batch_size, num_neurons).
        mem_rec: Tensor of shape (time_steps, batch_size, num_neurons).
        sample: batch index to visualize (default 0).
        output_neurons: iterable of output neuron indices to plot for membrane/spikes.
        figsize: figure size.
        palette: matplotlib palette name for line coloring.
        title: optional figure title.
        show: whether to call `plt.show()` before returning.

    Returns:
        (fig, [ax_membrane, ax_spikes])
    """
    # Normalize shapes if batch dim is missing
    if spk_rec.dim() == 2:
        spk_rec = spk_rec.unsqueeze(1)
        mem_rec = mem_rec.unsqueeze(1)

    T = spk_rec.size(0)

    # Extract sample
    mem2_rec = mem_rec[:, sample].cpu().numpy()
    spk2_rec = spk_rec[:, sample].cpu().numpy()

    # Choose neurons to plot
    def pick_indices(arr: np.ndarray, sel: Optional[Iterable[int]], default_max: int = 10):
        n = arr.shape[1]
        if sel is None:
            k = min(default_max, n)
            if k == 0:
                return []
            return list(range(k))
        return list(sel)

    out_idx = pick_indices(mem2_rec, output_neurons)

    sns.set(style="darkgrid")
    colors = sns.color_palette(palette, max(len(out_idx), 6))

    fig, (ax_mem, ax_spk) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1, 0.8]}
    )

    times = np.arange(T)

    # Membrane potentials
    if len(out_idx) > 0:
        for i, idx in enumerate(out_idx):
            ax_mem.plot(
                times, mem2_rec[:, idx], label=f"out{idx}", color=colors[i % len(colors)])
        ax_mem.set_ylabel("Membrane potential")
        ax_mem.legend(loc="upper right", ncol=min(3, len(out_idx)))
        ax_mem.axhline(0, color="0.6", linestyle="--")
    else:
        ax_mem.text(0.5, 0.5, "No output neurons to display",
                    ha="center", va="center")

    # Spikes raster
    if spk2_rec.size > 0 and spk2_rec.ndim == 2:
        spk_plot = spk2_rec[:, out_idx] if len(out_idx) > 0 else spk2_rec
        for i in range(spk_plot.shape[1]):
            times_i = np.where(spk_plot[:, i] > 0)[0]
            ax_spk.scatter(times_i, np.ones_like(times_i) * i,
                           marker="|", s=200, color=colors[i % len(colors)])

        ax_spk.set_ylabel("Neuron")
        ax_spk.set_xlabel("Timestep")
        ax_spk.set_yticks(range(spk_plot.shape[1]))
        ax_spk.set_yticklabels([f"n{idx}" for idx in (
            out_idx if len(out_idx) > 0 else range(spk_plot.shape[1]))])
    else:
        ax_spk.text(0.5, 0.5, "No spikes recorded", ha="center", va="center")

    if title is None:
        title = "SNN activity"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, [ax_mem, ax_spk]
