import torch
from torch.utils.data import TensorDataset, DataLoader
from moabb.paradigms import MotorImagery
import numpy as np


def get_motorimagery_loaders(
    dataset,
    sample_frequency=250,
    test_subject_id=1,
    n_subjects=-1,
    batch_size=32,
    split_mode="loso",
    test_size=0.2,
    random_state=42,
    preprocessing_fn=None
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Build train/test loaders for a dataset with two split modes.

    split_mode options:
    - "loso": leave one subject out for testing (test_subject_id).
    - "random": pool all subjects, shuffle, then split by test_size.

    """
    if split_mode not in {"loso", "random"}:
        raise ValueError("split_mode must be either 'loso' or 'random'")

    if not (0 < test_size < 1):
        raise ValueError("test_size must be in the interval (0, 1)")

    max_subjects = dataset.metadata.participants.n_subjects

    if n_subjects == -1:
        n_subjects = max_subjects

    if n_subjects > max_subjects:
        raise ValueError(
            f"{n_subjects} is more subjects than the avaiable {max_subjects}"
        )

    n_classes = dataset.metadata.experiment.n_classes
    paradigm = MotorImagery(n_classes=n_classes, fmin=4,
                            fmax=38, resample=sample_frequency)

    subjects = list(range(1, n_subjects + 1))
    X, y, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects)  # type: ignore

    # --- INJECT OPTIONAL PREPROCESSING ---
    if preprocessing_fn is not None:
        X, y, metadata = preprocessing_fn(X, y, metadata, sample_frequency)

    if split_mode == "loso":
        if test_subject_id not in subjects:
            raise ValueError(
                f"test_subject_id={test_subject_id} is not in selected subjects: {subjects}")

        subject_ids = metadata["subject"].to_numpy()
        test_mask = subject_ids == test_subject_id
        train_mask = ~test_mask

        if not np.any(test_mask):
            raise RuntimeError(
                f"No test trials found for subject {test_subject_id}. Check subject selection.")
        if not np.any(train_mask):
            raise RuntimeError("No training trials found after LOSO split.")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    else:
        # Pool all users, shuffle once, and split by ratio.
        indices = np.arange(len(y))
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

        n_test = int(len(indices) * test_size)
        n_test = min(max(n_test, 1), len(indices) - 1)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

    # Preprocessing: add EEGNet channel dimension -> (Batch, 1, Channels, Time)
    X_train = X_train[:, np.newaxis, :, :].astype(np.float32)
    X_test = X_test[:, np.newaxis, :, :].astype(np.float32)

    # Consistent label encoding
    unique_labels = np.unique(y)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    y_train = np.array([label_dict[l] for l in y_train])
    y_test = np.array([label_dict[l] for l in y_test])

    train_ds = TensorDataset(torch.from_numpy(
        X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(
        X_test), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("\n")
    print("="*20)
    if split_mode == "loso":
        print(f"Split Mode: LOSO | Test Subject: {test_subject_id}")
        print(
            f"Training on {len(X_train)} trials from {len(subjects) - 1} subjects")
        print(f"Testing on {len(X_test)} trials from 1 subject")
    else:
        print(
            f"Split Mode: RANDOM | Users pooled | test_size={test_size} | random_state={random_state}")
        print(f"Training on {len(X_train)} trials from all selected subjects")
        print(f"Testing on {len(X_test)} trials from all selected subjects")

    sample_data, _ = next(iter(train_loader))

    batch_size = sample_data.shape[0]
    electrode_channels = sample_data.shape[2]
    sample_length = sample_data.shape[3]

    print(f"Input Shape for Model: {sample_data.shape}")

    print(
        f"Batch size: {batch_size} | Nº Channels: {electrode_channels} | Sample length: {sample_length}")

    print("="*20)

    metadata = {
        "n_classes": n_classes,
        "electrode_channels": electrode_channels,
        "sample_length": sample_length
    }

    return train_loader, test_loader, metadata


def get_raw_data(dataset, n_subjects=-1, sample_frequency=128) -> tuple[np.ndarray, np.ndarray, dict]:
    max_subjects = dataset.metadata.participants.n_subjects

    if n_subjects == -1:
        n_subjects = max_subjects

    if n_subjects > max_subjects:
        raise ValueError(
            f"{n_subjects} is more subjects than the avaiable {max_subjects}"
        )

    n_classes = dataset.metadata.experiment.n_classes
    paradigm = MotorImagery(n_classes=n_classes, fmin=4,
                            fmax=38, resample=sample_frequency)

    subjects = list(range(1, n_subjects + 1))
    X, y, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects)  # type: ignore

    electrode_channels = X.shape[1]
    sample_length = X.shape[2]

    metadata_new = {
        "n_classes": n_classes,
        "electrode_channels": electrode_channels,
        "sample_length": sample_length,
        # Assuming metadata has a 'subject' column
        "subject": metadata.get("subject", [])
    }

    return X, y, metadata_new


def _get_loso(X, y, metadata, test_subject_id, **kwargs):
    subjects = np.unique(metadata["subject"])
    if test_subject_id not in subjects:
        raise ValueError(
            f"test_subject_id={test_subject_id} is not in selected subjects: {subjects}")

    subject_ids = metadata["subject"].to_numpy()
    test_mask = subject_ids == test_subject_id
    train_mask = ~test_mask

    if not np.any(test_mask):
        raise RuntimeError(
            f"No test trials found for subject {test_subject_id}. Check subject selection.")
    if not np.any(train_mask):
        raise RuntimeError("No training trials found after LOSO split.")

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def _get_random(X, y, test_size, random_state, **kwargs):
    indices = np.arange(len(y))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    n_test = int(len(indices) * test_size)
    n_test = min(max(n_test, 1), len(indices) - 1)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def _get_cross(X, y, metadata, test_size, random_state, **kwargs):
    subjects = np.unique(metadata["subject"])
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_test_subjects = max(int(len(subjects) * test_size), 1)
    test_subjects = subjects[:n_test_subjects]

    subject_ids = metadata["subject"].to_numpy()
    test_mask = np.isin(subject_ids, test_subjects)
    train_mask = ~test_mask

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def create_dataloaders(X, y, metadata, split_mode="loso", test_subject_id=1, test_size=0.2, random_state=42, batch_size=32, **kwargs):
    # Same as get_motorimagey_loaders but allows passing preprocessed data directly.
    # This is useful if you want to do custom preprocessing outside of the function.
    # The kwargs are ignored but allow for a consistent interface.

    split_funcs = {
        "loso": _get_loso,
        "random": _get_random,
        "cross": _get_cross
    }

    if split_mode not in split_funcs:
        raise ValueError(
            f"split_mode must be one of {list(split_funcs.keys())}")

    if not (0 < test_size < 1):
        raise ValueError("test_size must be in the interval (0, 1)")

    X_train, y_train, X_test, y_test = split_funcs[split_mode](
        X=X,
        y=y,
        metadata=metadata,
        test_subject_id=test_subject_id,
        test_size=test_size,
        random_state=random_state
    )

    # Preprocessing: add EEGNet channel dimension -> (Batch, 1, Channels, Time)
    X_train = X_train[:, np.newaxis, :, :].astype(np.float32)
    X_test = X_test[:, np.newaxis, :, :].astype(np.float32)

    # Consistent label encoding
    unique_labels = np.unique(y)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    y_train = np.array([label_dict[l] for l in y_train])
    y_test = np.array([label_dict[l] for l in y_test])

    train_ds = TensorDataset(torch.from_numpy(
        X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(
        X_test), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("\n")
    print("="*20)
    subjects = np.unique(metadata["subject"])
    if split_mode == "loso":
        print(
            f"Split Mode: {split_mode.upper()} | Test Subject: {test_subject_id}")
        print(
            f"Training on {len(X_train)} trials from {len(subjects) - 1} subjects")
        print(f"Testing on {len(X_test)} trials")
    elif split_mode == "cross":
        print(
            f"Split Mode: {split_mode.upper()} | test_size={test_size} | random_state={random_state}")
        print(f"Training on {len(X_train)} trials")
        print(f"Testing on {len(X_test)} trials")
    else:
        print(
            f"Split Mode: {split_mode.upper()} | Users pooled | test_size={test_size} | random_state={random_state}")
        print(f"Training on {len(X_train)} trials from all selected subjects")
        print(f"Testing on {len(X_test)} trials from all selected subjects")

    sample_data, _ = next(iter(train_loader))

    batch_size = sample_data.shape[0]
    electrode_channels = sample_data.shape[2]
    sample_length = sample_data.shape[3]

    print(f"Input Shape for Model: {sample_data.shape}")

    print(
        f"Batch size: {batch_size} | Nº Channels: {electrode_channels} | Sample length: {sample_length}")

    print("="*20)

    return train_loader, test_loader, metadata
