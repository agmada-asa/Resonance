from pathlib import Path
import os
import sys
from time import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.basic_model import SpectrogramTransitionModel
from experiments.dataset import SyntheticSpectrogramDataset
from experiments.unet_model import SpectrogramUNetModel

DATA_DIR = ROOT / "data/synthetic/v001"
BUILD_DIR = ROOT / "build"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 75
LEARNING_RATE = 1e-3

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


def main():
    from experiments.test import plot_sample

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
    # model = SpectrogramTransitionModel(action_vector_size).to(device)
    model = SpectrogramUNetModel(action_vector_size).to(device)

    # Define the loss function and optimizer
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
    # plot_sample(model, test_loader.dataset, device, sample_index=0)
    # plot_sample(model, test_loader.dataset, device, sample_index=1)
    # plot_sample(model, test_loader.dataset, device, sample_index=2)
    # plot_sample(model, test_loader.dataset, device, sample_index=3)
    # plot_sample(model, test_loader.dataset, device, sample_index=7)


if __name__ == "__main__":
    main()
