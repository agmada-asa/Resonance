from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from resonance.config import (
    ACTION_VECTOR_SIZE,
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    AudioConfig,
    ProjectPaths,
)
from resonance.data.cq_nsgt_cache import (
    DEFAULT_CQ_NSGT_CACHE_DIRNAME,
    CQNSGTShardBatchSampler,
    CachedCQNSGTDataset,
)
from resonance.models.cq_nsgt import CQNSGTTransitionModel


@dataclass(frozen=True)
class CQNSGTTrainingConfig:
    # CQ-NSGT has more values than the old CQT, so start with a smaller batch
    batch_size: int = 2
    num_epochs: int = 30
    learning_rate: float = 1e-3
    cache_dirname: str = DEFAULT_CQ_NSGT_CACHE_DIRNAME
    num_workers: int = 2
    best_checkpoint_name: str = 'best_cq_nsgt_transition_model.pth'
    latest_checkpoint_name: str = 'latest_cq_nsgt_transition_model.pth'
    final_checkpoint_name: str = 'cq_nsgt_transition_model_1.pth'


class CQNSGTTransitionTrainer:
    def __init__(
        self,
        training_config=CQNSGTTrainingConfig(),
        audio_config: AudioConfig = DEFAULT_CONFIG,
        paths: ProjectPaths = DEFAULT_PATHS,
    ):
        self.training_config = training_config
        self.audio_config = audio_config
        self.paths = paths

    # Create DataLoaders from the precomputed CQ-NSGT vector shards
    def create_dataloaders(self):
        data_dir = self.paths.cq_nsgt_data_dir
        cache_dir = data_dir / self.training_config.cache_dirname

        train_dataset = CachedCQNSGTDataset(
            cache_dir,
            'train',
            config=self.audio_config,
        )
        val_dataset = CachedCQNSGTDataset(
            cache_dir,
            'val',
            config=self.audio_config,
        )
        test_dataset = CachedCQNSGTDataset(
            cache_dir,
            'test',
            config=self.audio_config,
        )

        train_sampler = CQNSGTShardBatchSampler(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        val_sampler = CQNSGTShardBatchSampler(
            val_dataset,
            batch_size=self.training_config.batch_size,
        )
        test_sampler = CQNSGTShardBatchSampler(
            test_dataset,
            batch_size=self.training_config.batch_size,
        )
        loader_options = {
            'num_workers': self.training_config.num_workers,
            'persistent_workers': self.training_config.num_workers > 0,
        }
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            **loader_options,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            **loader_options,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            **loader_options,
        )

        return train_loader, val_loader, test_loader

    def move_batch_to_device(self, batch, device):
        # Coalesce all ragged blocks into one transfer. Sending 112 inputs and
        # 112 targets separately is particularly expensive on MPS because each
        # .to() call synchronizes through a small shape-specific graph.
        # Remember the ragged layout before joining the CPU tensors together
        input_shapes = [block.shape for block in batch['input']]
        target_shapes = [block.shape for block in batch['target']]
        input_sizes = [block.numel() for block in batch['input']]
        target_sizes = [block.numel() for block in batch['target']]
        input_value_count = sum(input_sizes)
        target_value_count = sum(target_sizes)
        action_shape = batch['action_vector'].shape

        # Include the action vector so the complete batch needs only one copy
        packed = torch.cat([
            *(block.reshape(-1) for block in batch['input']),
            *(block.reshape(-1) for block in batch['target']),
            batch['action_vector'].reshape(-1),
        ]).to(device)

        # Split the transferred storage into flat input, target, and action views
        input_flat = packed[:input_value_count]
        target_start = input_value_count
        target_stop = target_start + target_value_count
        target_flat = packed[target_start:target_stop]
        action_vector = packed[target_stop:].view(action_shape)

        # Restore zero-copy views with the shapes expected by the ragged model
        def restore_blocks(flat, shapes, sizes):
            blocks = []
            start = 0
            for shape, size in zip(shapes, sizes):
                blocks.append(flat[start:start + size].view(shape))
                start += size
            return blocks

        return {
            'input': restore_blocks(input_flat, input_shapes, input_sizes),
            'target': restore_blocks(target_flat, target_shapes, target_sizes),
            'input_flat': input_flat,
            'target_flat': target_flat,
            'action_vector': action_vector,
        }

    @staticmethod
    def print_progress(name, batch_index, total_batches, loss, start_time):
        elapsed = time() - start_time
        seconds_per_batch = elapsed / batch_index
        remaining = seconds_per_batch * (total_batches - batch_index)

        print(
            f'  {name} batch [{batch_index}/{total_batches}] '
            f'Loss: {loss:.6f} '
            f'Elapsed: {elapsed / 60:.1f}m '
            f'ETA: {remaining / 60:.1f}m',
            flush=True,
        )

    @staticmethod
    def block_mse(predicted_blocks, target_blocks):
        # Accept either normal block lists or an already packed target tensor
        if isinstance(predicted_blocks, torch.Tensor):
            predicted_flat = predicted_blocks.reshape(-1)
        else:
            predicted_flat = torch.cat([
                block.reshape(-1)
                for block in predicted_blocks
            ])
        if isinstance(target_blocks, torch.Tensor):
            target_flat = target_blocks.reshape(-1)
        else:
            if len(predicted_blocks) != len(target_blocks):
                raise ValueError(
                    'Predicted and target block counts do not match.'
                )
            target_flat = torch.cat([
                block.reshape(-1)
                for block in target_blocks
            ])
        # A flat comparison still requires the same total coefficient count
        if predicted_flat.shape != target_flat.shape:
            raise ValueError('Predicted and target value counts do not match.')

        # One flat reduction preserves equal coefficient weighting while
        # avoiding 112 separate MPS reductions and Python-side additions.
        return torch.mean((predicted_flat - target_flat) ** 2)

    def train_one_epoch(self, model, data_loader, optimizer, device):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_batches = len(data_loader)
        start_time = time()
        # Delay scalar reads so MPS can execute several batches asynchronously
        pending_losses = []

        for batch_index, batch in enumerate(data_loader, start=1):
            batch = self.move_batch_to_device(batch, device)

            # Predict the complete next CQ-NSGT vectors and update the model
            optimizer.zero_grad(set_to_none=True)
            predicted_target = model(
                batch['input'],
                batch['action_vector'],
            )
            loss = self.block_mse(predicted_target, batch['target_flat'])
            loss.backward()
            optimizer.step()

            batch_size = batch['action_vector'].shape[0]
            # Keep detached device scalars until the next progress report
            pending_losses.append(loss.detach() * batch_size)
            total_examples += batch_size

            # Print regularly without adding output for every individual batch
            if batch_index % 50 == 0 or batch_index == total_batches:
                total_loss += torch.stack(pending_losses).sum().item()
                pending_losses.clear()
                average_loss = total_loss / total_examples
                self.print_progress(
                    'Training',
                    batch_index,
                    total_batches,
                    average_loss,
                    start_time,
                )

        return total_loss / total_examples

    def evaluate(self, model, data_loader, device, name='Evaluation'):
        model.eval()
        total_target_loss = 0.0
        total_identity_loss = 0.0
        total_examples = 0
        total_batches = len(data_loader)
        start_time = time()
        # Validation uses the same delayed synchronization as training
        pending_target_losses = []
        pending_identity_losses = []

        with torch.no_grad():
            for batch_index, batch in enumerate(data_loader, start=1):
                batch = self.move_batch_to_device(batch, device)
                predicted_target = model(
                    batch['input'],
                    batch['action_vector'],
                )

                target_loss = self.block_mse(
                    predicted_target,
                    batch['target_flat'],
                )
                identity_loss = self.block_mse(
                    batch['input_flat'],
                    batch['target_flat'],
                )

                batch_size = batch['action_vector'].shape[0]
                pending_target_losses.append(target_loss * batch_size)
                pending_identity_losses.append(identity_loss * batch_size)
                total_examples += batch_size

                # Validation and testing also recreate CQ-NSGT vectors on demand
                if batch_index % 50 == 0 or batch_index == total_batches:
                    total_target_loss += (
                        torch.stack(pending_target_losses).sum().item()
                    )
                    total_identity_loss += (
                        torch.stack(pending_identity_losses).sum().item()
                    )
                    pending_target_losses.clear()
                    pending_identity_losses.clear()
                    average_loss = total_target_loss / total_examples
                    self.print_progress(
                        name,
                        batch_index,
                        total_batches,
                        average_loss,
                        start_time,
                    )

        return {
            'target_loss': total_target_loss / total_examples,
            'identity_loss': total_identity_loss / total_examples,
        }

    def select_device(self):
        # Use the fastest available training device
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def save_checkpoint(
        self,
        model,
        path,
        optimizer=None,
        completed_epochs=0,
        best_val_loss=float('inf'),
    ):
        path = Path(path)
        checkpoint = {
            'checkpoint_version': 2,
            'model_state_dict': model.state_dict(),
            'completed_epochs': completed_epochs,
            'best_val_loss': best_val_loss,
            'action_vector_size': ACTION_VECTOR_SIZE,
            'sample_rate': self.audio_config.sample_rate,
            'duration': self.audio_config.duration,
            'bins_per_octave': self.audio_config.bins_per_octave,
            'fmin': self.audio_config.fmin,
            'torch_rng_state': torch.get_rng_state(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()

        # Replace the destination only after serialization succeeds so an
        # interruption cannot leave the last usable checkpoint half-written.
        temporary_path = path.with_suffix(f'{path.suffix}.tmp')
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

    def _validate_checkpoint(self, checkpoint, path):
        expected_values = {
            'action_vector_size': ACTION_VECTOR_SIZE,
            'sample_rate': self.audio_config.sample_rate,
            'duration': self.audio_config.duration,
            'bins_per_octave': self.audio_config.bins_per_octave,
            'fmin': self.audio_config.fmin,
        }
        for name, expected_value in expected_values.items():
            actual_value = checkpoint.get(name)
            if actual_value != expected_value:
                raise ValueError(
                    f'Checkpoint {path} has incompatible {name}: '
                    f'{actual_value} != {expected_value}.'
                )

    def load_checkpoint(self, model, optimizer, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Resume checkpoint does not exist: {path}')

        checkpoint = torch.load(
            path,
            map_location='cpu',
            weights_only=True,
        )
        self._validate_checkpoint(checkpoint, path)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        torch_rng_state = checkpoint.get('torch_rng_state')
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)
        cuda_rng_state = checkpoint.get('cuda_rng_state')
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

        completed_epochs = checkpoint.get('completed_epochs', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return completed_epochs, best_val_loss, optimizer_state is not None

    def run(self, resume_from=None, start_epoch=None):
        device = self.select_device()
        print(f'Using device: {device}')

        train_loader, val_loader, test_loader = self.create_dataloaders()
        print(
            f'Training samples: {len(train_loader.dataset)} '
            f'({len(train_loader)} batches)'
        )
        print(
            f'Validation samples: {len(val_loader.dataset)} '
            f'({len(val_loader)} batches)'
        )
        print(
            f'Test samples: {len(test_loader.dataset)} '
            f'({len(test_loader)} batches)',
            flush=True,
        )

        model = CQNSGTTransitionModel(ACTION_VECTOR_SIZE).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
        )

        self.paths.build_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')
        completed_epochs = 0

        if resume_from is not None:
            (
                completed_epochs,
                best_val_loss,
                restored_optimizer,
            ) = self.load_checkpoint(model, optimizer, resume_from)
            if start_epoch is not None:
                completed_epochs = start_epoch

            restored_state = (
                'model and optimizer state'
                if restored_optimizer
                else 'model weights only (optimizer state was unavailable)'
            )
            print(
                f'Resumed {restored_state} from {resume_from}; '
                f'continuing at epoch {completed_epochs + 1}.',
                flush=True,
            )

            if best_val_loss == float('inf'):
                print(
                    'The checkpoint has no validation history; measuring its '
                    'validation loss before training resumes.',
                    flush=True,
                )
                resume_losses = self.evaluate(
                    model,
                    val_loader,
                    device,
                    name='Resume validation',
                )
                best_val_loss = resume_losses['target_loss']
                print(
                    f'Resumed checkpoint validation target loss: '
                    f'{best_val_loss:.6f}',
                    flush=True,
                )
        elif start_epoch is not None:
            raise ValueError('start_epoch requires a resume checkpoint.')

        if not 0 <= completed_epochs <= self.training_config.num_epochs:
            raise ValueError(
                'The number of completed epochs must be between 0 and '
                f'{self.training_config.num_epochs}, got {completed_epochs}.'
            )

        if resume_from is not None:
            # Immediately migrate legacy checkpoints to the complete format.
            # This also leaves an epoch-boundary fallback if the resumed run is
            # interrupted before it finishes its first new epoch.
            self.save_checkpoint(
                model,
                self.paths.build_dir
                / self.training_config.latest_checkpoint_name,
                optimizer=optimizer,
                completed_epochs=completed_epochs,
                best_val_loss=best_val_loss,
            )

        for epoch in range(completed_epochs, self.training_config.num_epochs):
            start_time = time()
            batch_sampler = getattr(train_loader, 'batch_sampler', None)
            if hasattr(batch_sampler, 'set_epoch'):
                batch_sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )
            val_losses = self.evaluate(
                model,
                val_loader,
                device,
                name='Validation',
            )

            # Keep the checkpoint with the best validation target loss
            if val_losses['target_loss'] < best_val_loss:
                best_val_loss = val_losses['target_loss']
                self.save_checkpoint(
                    model,
                    self.paths.build_dir
                    / self.training_config.best_checkpoint_name,
                    optimizer=optimizer,
                    completed_epochs=epoch + 1,
                    best_val_loss=best_val_loss,
                )

            print(
                f'Epoch [{epoch + 1}/{self.training_config.num_epochs}] '
                f'Train target loss: {train_loss:.6f} '
                f"Val target loss: {val_losses['target_loss']:.6f} "
                f"Val identity baseline: {val_losses['identity_loss']:.6f} "
                f'Time: {time() - start_time:.2f}s'
            )

            # Always retain the most recently completed epoch for resuming.
            self.save_checkpoint(
                model,
                self.paths.build_dir
                / self.training_config.latest_checkpoint_name,
                optimizer=optimizer,
                completed_epochs=epoch + 1,
                best_val_loss=best_val_loss,
            )

        test_losses = self.evaluate(
            model,
            test_loader,
            device,
            name='Testing',
        )
        print(
            f"Test target loss: {test_losses['target_loss']:.6f} "
            f"Test identity baseline: {test_losses['identity_loss']:.6f}"
        )

        self.save_checkpoint(
            model,
            self.paths.build_dir
            / self.training_config.final_checkpoint_name,
            optimizer=optimizer,
            completed_epochs=self.training_config.num_epochs,
            best_val_loss=best_val_loss,
        )

        return {
            'model': model,
            'test_losses': test_losses,
            'test_dataset': test_loader.dataset,
        }
