import numpy as np
import mne
from sklearn.decomposition import FastICA


def apply_epoch_ica(X: np.ndarray, y: np.ndarray, metadata, sfreq: float, n_components: int = 15):
    """
    Applies FastICA to epoched raw data (Trials, Channels, Time) manually via sklearn.
    This demonstrates how to inject custom processing (e.g. dropping ICs automatically
    or denoising) before the DataLoader splits the dataset.
    """
    print(f"Applying FastICA (n_components={n_components}) to epoched data...")
    n_trials, n_channels, n_times = X.shape

    # Reshape to (Samples, Channels) where Samples = n_trials * n_times
    # Best practice for epoch-based ICA is to stack all trials together
    X_flat = np.transpose(X, (1, 0, 2)).reshape(n_channels, -1).T

    ica = FastICA(n_components=n_components, random_state=42)
    S_ = ica.fit_transform(X_flat)

    # Future custom logic goes here: identify noise components in S_
    # (e.g. by correlating with EOG channels if they exist) and set them to zero.
    # Ex: S_[:, artifact_indices] = 0

    # Reconstruct the cleaned signal
    X_reconstructed = ica.inverse_transform(S_)

    # Reshape back to (Trials, Channels, Time)
    X_clean = X_reconstructed.T.reshape(
        n_channels, n_trials, n_times).transpose(1, 0, 2)

    return X_clean.astype(np.float32), y, metadata


def apply_mne_filtering(X: np.ndarray, y: np.ndarray, metadata, sfreq: float, l_freq: float = 4.0, h_freq: float = 38.0):
    """
    Applies an MNE-based temporal bandpass filter onto the epoched data.
    MOABB Paradigms usually do this, but this is an example of injecting MNE steps.
    """
    print(f"Applying custom MNE Bandpass Filter ({l_freq}-{h_freq} Hz)...")
    X_clean = mne.filter.filter_data(
        X.astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
    return X_clean.astype(np.float32), y, metadata


def apply_mne_ica(X: np.ndarray, y: np.ndarray, metadata, sfreq: float, n_components: int = 25, method: str = "fastica", max_iter: int = 1000):

    print(
        f"Applying MNE ICA (method={method}, n_components={n_components}) to epoched data...")

    info = mne.create_info(X.shape[1], sfreq=sfreq, ch_types='eeg')

    # The paradigm filters the data between 4Hz and 38Hz before it gets here.
    # Manually tell MNE that this data has already been high-pass filtered to suppress warnings.
    with info._unlock():
        info['highpass'] = 4.0

    epochs = mne.EpochsArray(X, info, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter  # type: ignore
    )

    ica.fit(epochs)
    epochs_clean = ica.apply(epochs)
    X_clean = epochs_clean.get_data(copy=False)

    return X_clean, y, metadata
