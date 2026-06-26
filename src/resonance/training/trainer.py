from dataclasses import dataclass
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resonance.config import DEFAULT_PATHS, ProjectPaths
from resonance.data.dataset import SyntheticSpectrogramDataset
from resonance.models.basic import SpectrogramTransitionModel
from resonance.models.unet import SpectrogramUNetModel


@dataclass(frozen=True)
class TrainingConfig:
    # Hyperparameters
    batch_size: int = 32
    num_epochs: int = 75
    learning_rate: float = 1e-3
    train_filename: str = "train.npz"
    val_filename: str = "val.npz"
    test_filename: str = "test.npz"
    best_checkpoint_name: str = "best_spectrogram_transition_model.pth"
    final_checkpoint_name: str = "spectrogram_transition_model_1.pth"
    use_unet: bool = True


class SpectrogramTransitionTrainer:
    def __init__(
        self,
        training_config: TrainingConfig = TrainingConfig(),
        paths: ProjectPaths = DEFAULT_PATHS,
    ):
        self.training_config = training_config
        self.paths = paths

    # Get the mean and standard deviation of the training spectrograms for normalization
    def load_normalization_stats(self, train_path):
        with np.load(train_path) as data:
            train_inputs = data["input_spectrograms"].astype(np.float32)

        mean = float(train_inputs.mean())
        std = float(train_inputs.std())
        if std == 0:
            raise ValueError("Training spectrogram standard deviation is zero.")

        return mean, std

    # Create DataLoaders for training, validation, and testing
    def create_dataloaders(self, mean, std):
        # Use the same mean and std to normalize the spectrograms in the datasets
        train_dataset = SyntheticSpectrogramDataset(self.paths.data_dir / self.training_config.train_filename, mean, std)
        val_dataset = SyntheticSpectrogramDataset(self.paths.data_dir / self.training_config.val_filename, mean, std)
        test_dataset = SyntheticSpectrogramDataset(self.paths.data_dir / self.training_config.test_filename, mean, std)

        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.training_config.batch_size)

        return train_loader, val_loader, test_loader

    def move_batch_to_device(self, batch, device):
        return {key: value.to(device) for key, value in batch.items()}

    def train_one_epoch(self, model, data_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        total_examples = 0

        # Loop over the batches in the data loader
        for batch in data_loader:
            # Move the batch to the specified device (CPU or GPU)
            batch = self.move_batch_to_device(batch, device)

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

    def evaluate(self, model, data_loader, criterion, device):
        # Evaluate the model on the validation or test set without updating the model parameters
        model.eval()
        total_delta_loss = 0.0
        total_target_loss = 0.0
        total_identity_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch, device)

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

    def select_device(self):
        # Set the device to GPU if available, otherwise use CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def build_model(self, action_vector_size, device):
        if self.training_config.use_unet:
            return SpectrogramUNetModel(action_vector_size).to(device)

        return SpectrogramTransitionModel(action_vector_size).to(device)

    def save_checkpoint(self, model, mean, std, action_vector_size, path):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "action_vector_size": action_vector_size,
            },
            path,
        )

    def run(self, plot_sample_indices=None):
        device = self.select_device()

        print(f"Using device: {device}")
        mean, std = self.load_normalization_stats(self.paths.data_dir / self.training_config.train_filename)
        train_loader, val_loader, test_loader = self.create_dataloaders(mean, std)

        action_vector_size = train_loader.dataset.action_vectors.shape[1]
        # model = SpectrogramTransitionModel(action_vector_size).to(device)
        model = self.build_model(action_vector_size, device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.learning_rate)

        self.paths.build_dir.mkdir(parents=True, exist_ok=True)
        best_val_delta_loss = float("inf")

        # Train the model for the specified number of epochs, printing the training and validation losses after each epoch
        for epoch in range(self.training_config.num_epochs):
            start_time = time()
            train_loss = self.train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_losses = self.evaluate(model, val_loader, criterion, device)

            # Save the model checkpoint if it achieves the best validation delta loss so far
            if val_losses["delta_loss"] < best_val_delta_loss:
                best_val_delta_loss = val_losses["delta_loss"]
                self.save_checkpoint(
                    model,
                    mean,
                    std,
                    action_vector_size,
                    self.paths.build_dir / self.training_config.best_checkpoint_name,
                )

            print(
                f"Epoch [{epoch + 1}/{self.training_config.num_epochs}] "
                f"Train delta loss: {train_loss:.4f} "
                f"Val delta loss: {val_losses['delta_loss']:.4f} "
                f"Val target loss: {val_losses['target_loss']:.4f} "
                f"Val identity baseline: {val_losses['identity_loss']:.4f} "
                f"Time: {time() - start_time:.2f}s"
            )

        test_losses = self.evaluate(model, test_loader, criterion, device)
        print(
            f"Test delta loss: {test_losses['delta_loss']:.4f} "
            f"Test target loss: {test_losses['target_loss']:.4f} "
            f"Test identity baseline: {test_losses['identity_loss']:.4f}"
        )

        # Save the trained model, normalization statistics, and action vector size to the build directory
        self.save_checkpoint(
            model,
            mean,
            std,
            action_vector_size,
            self.paths.build_dir / self.training_config.final_checkpoint_name,
        )

        # Plot a sample from the test dataset to visualize the input, predicted target, and true target spectrograms
        if plot_sample_indices:
            from resonance.evaluation.evaluator import SpectrogramEvaluator

            evaluator = SpectrogramEvaluator()
            for sample_index in plot_sample_indices:
                evaluator.plot_sample(model, test_loader.dataset, device, sample_index=sample_index)

        return {
            "model": model,
            "mean": mean,
            "std": std,
            "test_losses": test_losses,
            "test_dataset": test_loader.dataset,
        }


DEFAULT_TRAINER = SpectrogramTransitionTrainer()


def load_normalization_stats(train_path):
    return DEFAULT_TRAINER.load_normalization_stats(train_path)


def create_dataloaders(mean, std):
    return DEFAULT_TRAINER.create_dataloaders(mean, std)


def move_batch_to_device(batch, device):
    return DEFAULT_TRAINER.move_batch_to_device(batch, device)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    return DEFAULT_TRAINER.train_one_epoch(model, data_loader, criterion, optimizer, device)


def evaluate(model, data_loader, criterion, device):
    return DEFAULT_TRAINER.evaluate(model, data_loader, criterion, device)

