import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/synthetic/v001"
BUILD_DIR = ROOT / "build"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

class SyntheticSpectrogramDataset(Dataset):
    def __init__(self, npz_path, mean, std):
        # Load the dataset from the .npz file and normalize the spectrograms
        with np.load(npz_path) as data:
            input_spectrograms = data["input_spectrograms"].astype(np.float32)
            target_spectrograms = data["target_spectrograms"].astype(np.float32)
            action_vectors = data["action_vectors"].astype(np.float32)

        input_spectrograms = (input_spectrograms - mean) / std
        target_spectrograms = (target_spectrograms - mean) / std
        target_deltas = target_spectrograms - input_spectrograms

        self.inputs = torch.from_numpy(input_spectrograms).unsqueeze(1)
        self.target_deltas = torch.from_numpy(target_deltas).unsqueeze(1)
        self.targets = torch.from_numpy(target_spectrograms).unsqueeze(1)
        self.action_vectors = torch.from_numpy(action_vectors)
        self.metadata = self._load_metadata(npz_path)

    @staticmethod
    def _load_metadata(npz_path):
        metadata_path = npz_path.with_name(f"metadata_{npz_path.stem}.jsonl")
        if not metadata_path.exists():
            return None

        with metadata_path.open() as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return {
            "input": self.inputs[index],
            "target_delta": self.target_deltas[index],
            "target": self.targets[index],
            "action_vector": self.action_vectors[index],
        }


class SpectrogramTransitionModel(nn.Module):
    def __init__(self, action_vector_size, action_embedding_size=32):
        super().__init__()
        # Define the encoder, action encoder, and decoder networks

        # Encoder is a simple CNN that processes the input spectrogram
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        # Action encoder is a simple feedforward network that processes the action vector
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, 64),
            nn.GELU(),
            nn.Linear(64, action_embedding_size),
            nn.GELU(),
        )

        # Decoder is a simple CNN that reconstructs the output spectrogram from the encoded features and action embedding
        self.decoder = nn.Sequential(
            nn.Conv2d(64 + action_embedding_size, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, action_vector):
        input_shape = x.shape[-2:]
        features = self.encoder(x)

        # Create an action embedding and expand it to match the spatial dimensions of the features
        action_embedding = self.action_encoder(action_vector)
        action_embedding = action_embedding[:, :, None, None]
        action_embedding = action_embedding.expand(
            -1,
            -1,
            features.shape[-2],
            features.shape[-1],
        )
        
        # Concatenate the features and action embedding along the channel dimension
        conditioned_features = torch.cat((features, action_embedding), dim=1)

        # Pass the conditioned features through the decoder to predict the delta spectrogram
        predicted_delta = self.decoder(conditioned_features)
        return F.interpolate(
            predicted_delta,
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )

# Get the mean and standard deviation of the training spectrograms for normalization
def load_normalization_stats(train_path):
    with np.load(train_path) as data:
        train_inputs = data["input_spectrograms"].astype(np.float32)

    mean = float(train_inputs.mean())
    std = float(train_inputs.std())
    if std == 0:
        raise ValueError("Training spectrogram standard deviation is zero.")

    return mean, std

# Create DataLoaders for training, validation, and testing
def create_dataloaders(mean, std):
    # Use the same mean and std to normalize the spectrograms in the datasets
    train_dataset = SyntheticSpectrogramDataset(DATA_DIR / "train.npz", mean, std)
    val_dataset = SyntheticSpectrogramDataset(DATA_DIR / "val.npz", mean, std)
    test_dataset = SyntheticSpectrogramDataset(DATA_DIR / "test.npz", mean, std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    # Loop over the batches in the data loader
    for batch in data_loader:
        # Move the batch to the specified device (CPU or GPU)
        batch = move_batch_to_device(batch, device)

        # Zero the gradients, perform a forward pass, compute the loss, perform a backward pass, and update the model parameters
        optimizer.zero_grad()
        predicted_delta = model(batch["input"], batch["action_vector"])
        loss = criterion(predicted_delta, batch["target_delta"])
        loss.backward()
        optimizer.step()

        # Update the total loss and total number of examples processed
        batch_size = batch["input"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    # Return the average loss for the epoch
    return total_loss / total_examples


def evaluate(model, data_loader, criterion, device):
    # Evaluate the model on the validation or test set without updating the model parameters
    model.eval()
    total_delta_loss = 0.0
    total_target_loss = 0.0
    total_identity_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)

            # Forward pass to get the predicted delta and predicted target spectrograms
            predicted_delta = model(batch["input"], batch["action_vector"])
            predicted_target = batch["input"] + predicted_delta

            delta_loss = criterion(predicted_delta, batch["target_delta"])
            target_loss = criterion(predicted_target, batch["target"])
            identity_loss = criterion(batch["input"], batch["target"])

            batch_size = batch["input"].shape[0]
            total_delta_loss += delta_loss.item() * batch_size
            total_target_loss += target_loss.item() * batch_size
            total_identity_loss += identity_loss.item() * batch_size
            total_examples += batch_size

    return {
        "delta_loss": total_delta_loss / total_examples,
        "target_loss": total_target_loss / total_examples,
        "identity_loss": total_identity_loss / total_examples,
    }

# Plot a sample from the dataset, showing the input spectrogram, predicted target spectrogram, and true target spectrogram
def plot_sample(model, dataset, device, sample_index=0):
    import matplotlib.pyplot as plt

    sample = dataset[sample_index]
    metadata = dataset.metadata[sample_index] if dataset.metadata else None
    sample_input = sample["input"].unsqueeze(0).to(device)
    sample_action_vector = sample["action_vector"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predicted_delta = model(sample_input, sample_action_vector)
        predicted_target = sample_input + predicted_delta

    input_spectrogram = sample["input"].squeeze().cpu().numpy()
    predicted_spectrogram = predicted_target.squeeze().cpu().numpy()
    target_spectrogram = sample["target"].squeeze().cpu().numpy()
    predicted_delta_spectrogram = predicted_delta.squeeze().cpu().numpy()
    target_delta_spectrogram = sample["target_delta"].squeeze().cpu().numpy()
    delta_error = predicted_delta_spectrogram - target_delta_spectrogram

    spectrogram_vmin = min(
        input_spectrogram.min(),
        predicted_spectrogram.min(),
        target_spectrogram.min(),
    )
    spectrogram_vmax = max(
        input_spectrogram.max(),
        predicted_spectrogram.max(),
        target_spectrogram.max(),
    )
    delta_abs_max = max(
        abs(predicted_delta_spectrogram).max(),
        abs(target_delta_spectrogram).max(),
        abs(delta_error).max(),
    )

    if metadata:
        print(
            "Sample metadata: "
            f"action={metadata['action']}, "
            f"parameter={metadata['parameter']}, "
            f"waveform={metadata['waveform_type']}, "
            f"frequency={metadata['frequency']:.2f}Hz"
        )

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Input Spectrogram")
    plt.imshow(
        input_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 2)
    plt.title("Predicted Target")
    plt.imshow(
        predicted_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 3)
    plt.title("True Target")
    plt.imshow(
        target_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 4)
    plt.title("Predicted Delta")
    plt.imshow(
        predicted_delta_spectrogram,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    plt.subplot(2, 3, 5)
    plt.title("True Delta")
    plt.imshow(
        target_delta_spectrogram,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    plt.subplot(2, 3, 6)
    plt.title("Delta Error")
    plt.imshow(
        delta_error,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    plt.tight_layout()
    plt.show()


def main():
    # Set the device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    mean, std = load_normalization_stats(DATA_DIR / "train.npz")
    train_loader, val_loader, test_loader = create_dataloaders(mean, std)

    action_vector_size = train_loader.dataset.action_vectors.shape[1]
    model = SpectrogramTransitionModel(action_vector_size).to(device)

    # Define the loss function (MSE) and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    best_val_delta_loss = float("inf")

    # Train the model for the specified number of epochs, printing the training and validation losses after each epoch
    for epoch in range(NUM_EPOCHS):
        start_time = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = evaluate(model, val_loader, criterion, device)

        # Save the model checkpoint if it achieves the best validation delta loss so far
        if val_losses["delta_loss"] < best_val_delta_loss:
            best_val_delta_loss = val_losses["delta_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "action_vector_size": action_vector_size,
                },
                BUILD_DIR / "best_spectrogram_transition_model.pth",
            )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train delta loss: {train_loss:.4f} "
            f"Val delta loss: {val_losses['delta_loss']:.4f} "
            f"Val target loss: {val_losses['target_loss']:.4f} "
            f"Val identity baseline: {val_losses['identity_loss']:.4f} "
            f"Time: {time() - start_time:.2f}s"
        )

    test_losses = evaluate(model, test_loader, criterion, device)
    print(
        f"Test delta loss: {test_losses['delta_loss']:.4f} "
        f"Test target loss: {test_losses['target_loss']:.4f} "
        f"Test identity baseline: {test_losses['identity_loss']:.4f}"
    )

    # Save the trained model, normalization statistics, and action vector size to the build directory
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "action_vector_size": action_vector_size,
        },
        BUILD_DIR / "spectrogram_transition_model_1.pth",
    )

    # Plot a sample from the test dataset to visualize the input, predicted target, and true target spectrograms
    plot_sample(model, test_loader.dataset, device, sample_index=0)
    plot_sample(model, test_loader.dataset, device, sample_index=1)
    plot_sample(model, test_loader.dataset, device, sample_index=2)
    plot_sample(model, test_loader.dataset, device, sample_index=3)
    plot_sample(model, test_loader.dataset, device, sample_index=7)


if __name__ == "__main__":
    main()
