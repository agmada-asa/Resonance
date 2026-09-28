import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from resonance.config import ACTION_VECTOR_SIZE, ProjectPaths
from resonance.data.cq_nsgt_cache import (
    CQNSGTShardBatchSampler,
    CachedCQNSGTDataset,
    precompute_cq_nsgt_cache,
)
from resonance.models.cq_nsgt import CQNSGTTransitionModel, resize_time
from resonance.models.cq_nsgt_global import CQNSGTGlobalTransitionModel
from resonance.training.cq_nsgt_trainer import (
    CQNSGTTrainingConfig,
    CQNSGTTransitionTrainer,
)


class CQNSGTTransitionModelTests(unittest.TestCase):
    def test_ragged_resize_matches_independent_pytorch_interpolation(self):
        torch.manual_seed(81)
        source = torch.randn(2, 3, 2, 5, requires_grad=True)
        reference_source = source.detach().clone().requires_grad_()
        actual = resize_time(
            source, torch.tensor([3, 5]), torch.tensor([5, 9]), 9,
        )
        expected = torch.cat([
            torch.nn.functional.pad(
                torch.nn.functional.interpolate(
                    reference_source[index:index + 1, :, :, :input_length],
                    size=(2, output_length), mode='bilinear', align_corners=False,
                ),
                (0, 9 - output_length),
            )
            for index, (input_length, output_length) in enumerate(((3, 5), (5, 9)))
        ])
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(source.grad, reference_source.grad)

    def test_cached_layouts_preserve_predictions_and_training_gradients(self):
        torch.manual_seed(73)
        reference = CQNSGTTransitionModel(
            ACTION_VECTOR_SIZE, cache_layouts=False,
        )
        cached = CQNSGTTransitionModel(ACTION_VECTOR_SIZE)
        cached.load_state_dict(reference.state_dict(), strict=True)

        # Exercise odd lengths, padded buckets, and cache hits on a second step
        for batch_size in (2, 2, 1):
            reference.zero_grad(set_to_none=True)
            cached.zero_grad(set_to_none=True)
            blocks = [
                torch.randn(batch_size, 2, height, length, requires_grad=True)
                for height, length in ((1, 5), (1, 6), (1, 8), (2, 7), (2, 10))
            ]
            cached_blocks = [b.detach().clone().requires_grad_() for b in blocks]
            actions = torch.randn(batch_size, ACTION_VECTOR_SIZE)
            expected = reference(blocks, actions)
            actual = cached(cached_blocks, actions)
            for expected_block, actual_block in zip(expected, actual):
                torch.testing.assert_close(actual_block, expected_block)
            sum(b.square().mean() for b in expected).backward()
            sum(b.square().mean() for b in actual).backward()
            for old, new in zip(reference.parameters(), cached.parameters()):
                torch.testing.assert_close(new.grad, old.grad, rtol=1e-5, atol=1e-6)
            for old, new in zip(blocks, cached_blocks):
                torch.testing.assert_close(new.grad, old.grad, rtol=1e-5, atol=1e-6)

        # Layout tensors must not change checkpoint keys or survive a dtype move
        self.assertEqual(set(reference.state_dict()), set(cached.state_dict()))
        cached.double()
        self.assertFalse(cached._layout_cache)
        actual = cached([b.detach().double() for b in blocks], actions.double())
        expected = reference.double()(
            [b.detach().double() for b in blocks], actions.double(),
        )
        for expected_block, actual_block in zip(expected, actual):
            torch.testing.assert_close(actual_block, expected_block)

    def test_grouped_blocks_match_separate_model_calls(self):
        torch.manual_seed(42)
        model = CQNSGTTransitionModel(ACTION_VECTOR_SIZE).eval()
        action_vector = torch.zeros(1, ACTION_VECTOR_SIZE)

        # Use different CQ-NSGT-style frequency and time resolutions
        vector_blocks = [
            torch.randn(1, 2, 1, 5),
            torch.randn(1, 2, 1, 6),
            torch.randn(1, 2, 1, 8),
            torch.randn(1, 2, 2, 7),
            torch.randn(1, 2, 2, 10),
        ]

        with torch.no_grad():
            separate = model._forward_blocks_separately(
                vector_blocks,
                action_vector,
            )
            grouped = model(vector_blocks, action_vector)

        self.assertEqual(len(grouped), len(vector_blocks))
        for source, separate_output, grouped_output in zip(
            vector_blocks,
            separate,
            grouped,
        ):
            self.assertEqual(grouped_output.shape, source.shape)
            torch.testing.assert_close(
                grouped_output,
                separate_output,
                rtol=1e-5,
                atol=1e-5,
            )


class CQNSGTCacheTests(unittest.TestCase):
    @staticmethod
    def _fake_dataset(metadata_path, config):
        class FakeDataset:
            def __init__(self):
                with Path(metadata_path).open() as source:
                    self.metadata = [json.loads(line) for line in source]

            def __len__(self):
                return len(self.metadata)

            def __getitem__(self, index):
                value = float(index)
                return {
                    'input': [
                        torch.full((2, 1, 4), value),
                        torch.full((2, 2, 5), value + 0.25),
                    ],
                    'target': [
                        torch.full((2, 1, 4), value + 1),
                        torch.full((2, 2, 5), value + 1.25),
                    ],
                    'action_vector': torch.full(
                        (ACTION_VECTOR_SIZE,),
                        value,
                    ),
                }

        return FakeDataset()

    def test_precompute_writes_loadable_shards_without_target_delta(self):
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            metadata_path = data_dir / 'metadata_train.jsonl'
            with metadata_path.open('w') as destination:
                for index in range(5):
                    destination.write(json.dumps({'index': index}) + '\n')

            with patch(
                'resonance.data.cq_nsgt_cache.SyntheticCQNSGTDataset',
                side_effect=self._fake_dataset,
            ):
                manifest = precompute_cq_nsgt_cache(
                    data_dir,
                    shard_size=2,
                    splits=('train',),
                )

            self.assertTrue(manifest['complete'])
            self.assertEqual(len(manifest['splits']['train']['shards']), 3)
            dataset = CachedCQNSGTDataset(
                data_dir / 'vector_cache_v1',
                'train',
            )
            self.assertEqual(len(dataset), 5)
            sample = dataset[3]
            self.assertNotIn('target_delta', sample)
            torch.testing.assert_close(
                sample['input'][0],
                torch.full((2, 1, 4), 3.0),
            )
            torch.testing.assert_close(
                sample['target'][1],
                torch.full((2, 2, 5), 4.25),
            )

    def test_shard_sampler_visits_every_sample_and_keeps_batches_local(self):
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            metadata_path = data_dir / 'metadata_train.jsonl'
            with metadata_path.open('w') as destination:
                for index in range(7):
                    destination.write(json.dumps({'index': index}) + '\n')

            with patch(
                'resonance.data.cq_nsgt_cache.SyntheticCQNSGTDataset',
                side_effect=self._fake_dataset,
            ):
                precompute_cq_nsgt_cache(
                    data_dir,
                    shard_size=3,
                    splits=('train',),
                )

            dataset = CachedCQNSGTDataset(
                data_dir / 'vector_cache_v1',
                'train',
            )
            sampler = CQNSGTShardBatchSampler(
                dataset,
                batch_size=2,
                shuffle=True,
                seed=42,
            )
            sampler.set_epoch(3)
            batches = list(sampler)
            flattened = [index for batch in batches for index in batch]
            self.assertEqual(sorted(flattened), list(range(7)))
            for batch in batches:
                shard_indices = {
                    next(
                        shard_index
                        for shard_index, descriptor in enumerate(dataset.shards)
                        if descriptor['start'] <= index
                        < descriptor['start'] + descriptor['length']
                    )
                    for index in batch
                }
                self.assertEqual(len(shard_indices), 1)

    def test_trainer_dataloaders_collate_cached_blocks(self):
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for split in ('train', 'val', 'test'):
                metadata_path = data_dir / f'metadata_{split}.jsonl'
                with metadata_path.open('w') as destination:
                    for index in range(3):
                        destination.write(json.dumps({'index': index}) + '\n')

            with patch(
                'resonance.data.cq_nsgt_cache.SyntheticCQNSGTDataset',
                side_effect=self._fake_dataset,
            ):
                precompute_cq_nsgt_cache(data_dir, shard_size=2)

            trainer = CQNSGTTransitionTrainer(
                training_config=CQNSGTTrainingConfig(
                    batch_size=2,
                    num_workers=2,
                ),
                paths=ProjectPaths(cq_nsgt_data_dir=data_dir),
            )
            train_loader, val_loader, test_loader = trainer.create_dataloaders()
            self.assertEqual(len(train_loader), 2)
            self.assertEqual(len(val_loader), 2)
            self.assertEqual(len(test_loader), 2)
            batch = next(iter(train_loader))
            self.assertEqual(batch['action_vector'].shape, (2, ACTION_VECTOR_SIZE))
            self.assertEqual(batch['input'][0].shape, (2, 2, 1, 4))
            self.assertEqual(batch['target'][1].shape, (2, 2, 2, 5))
            self.assertNotIn('target_delta', batch)

            moved = trainer.move_batch_to_device(batch, torch.device('cpu'))
            expected_input_values = sum(
                block.numel()
                for block in batch['input']
            )
            expected_target_values = sum(
                block.numel()
                for block in batch['target']
            )
            self.assertEqual(moved['input_flat'].numel(), expected_input_values)
            self.assertEqual(moved['target_flat'].numel(), expected_target_values)
            for original, restored in zip(batch['input'], moved['input']):
                torch.testing.assert_close(restored, original)
            for original, restored in zip(batch['target'], moved['target']):
                torch.testing.assert_close(restored, original)
            list_loss = trainer.block_mse(moved['input'], moved['target'])
            flat_loss = trainer.block_mse(
                moved['input_flat'],
                moved['target_flat'],
            )
            torch.testing.assert_close(list_loss, flat_loss)


class CQNSGTGlobalTransitionModelTests(unittest.TestCase):
    def test_no_action_starts_as_an_exact_identity(self):
        torch.manual_seed(42)
        model = CQNSGTGlobalTransitionModel().eval()
        action_vector = torch.zeros(1, ACTION_VECTOR_SIZE)

        # The real codec has 122 ordered bands split across ragged blocks
        vector_blocks = [torch.randn(1, 2, 122, 8)]

        with torch.no_grad():
            predicted = model(vector_blocks, action_vector)

        torch.testing.assert_close(
            predicted[0],
            vector_blocks[0],
            rtol=0,
            atol=0,
        )

    def test_butterworth_controllers_attenuate_the_correct_side(self):
        model = CQNSGTGlobalTransitionModel().eval()

        # Zero is the encoded midpoint between 20 Hz and 20 kHz
        cutoff = torch.zeros(1)
        low_pass = model._butterworth_response(cutoff, high_pass=False)
        high_pass = model._butterworth_response(cutoff, high_pass=True)
        frequencies = model.band_frequencies
        low_frequency = torch.argmin(torch.abs(frequencies - 100))
        high_frequency = torch.argmin(torch.abs(frequencies - 10000))

        self.assertGreater(
            torch.abs(low_pass[0, low_frequency]).item(),
            torch.abs(low_pass[0, high_frequency]).item(),
        )
        self.assertLess(
            torch.abs(high_pass[0, low_frequency]).item(),
            torch.abs(high_pass[0, high_frequency]).item(),
        )


class CQNSGTCheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_training_state(self):
        with TemporaryDirectory() as temporary_directory:
            paths = ProjectPaths(
                build_dir=Path(temporary_directory),
            )
            trainer = CQNSGTTransitionTrainer(paths=paths)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Populate Adam's momentum buffers before saving.
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            expected_parameters = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            checkpoint_path = paths.build_dir / 'checkpoint.pth'
            trainer.save_checkpoint(
                model,
                checkpoint_path,
                optimizer=optimizer,
                completed_epochs=4,
                best_val_loss=0.125,
            )

            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(),
                lr=0.5,
            )
            result = trainer.load_checkpoint(
                restored_model,
                restored_optimizer,
                checkpoint_path,
            )

            self.assertEqual(result, (4, 0.125, True))
            for name, value in restored_model.state_dict().items():
                torch.testing.assert_close(value, expected_parameters[name])
            self.assertEqual(
                len(restored_optimizer.state_dict()['state']),
                len(optimizer.state_dict()['state']),
            )

    def test_legacy_weight_only_checkpoint_can_be_loaded(self):
        with TemporaryDirectory() as temporary_directory:
            paths = ProjectPaths(
                build_dir=Path(temporary_directory),
            )
            trainer = CQNSGTTransitionTrainer(paths=paths)
            model = torch.nn.Linear(2, 1)
            checkpoint_path = paths.build_dir / 'legacy.pth'
            trainer.save_checkpoint(model, checkpoint_path)

            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=True,
            )
            checkpoint.pop('checkpoint_version')
            checkpoint.pop('completed_epochs')
            checkpoint.pop('best_val_loss')
            checkpoint.pop('torch_rng_state')
            torch.save(checkpoint, checkpoint_path)

            optimizer = torch.optim.Adam(model.parameters())
            self.assertEqual(
                trainer.load_checkpoint(model, optimizer, checkpoint_path),
                (0, float('inf'), False),
            )

    def test_resumed_run_writes_latest_full_checkpoint(self):
        class FakeLoader:
            dataset = [object()]

            def __len__(self):
                return 1

        class FastTrainer(CQNSGTTransitionTrainer):
            def select_device(self):
                return torch.device('cpu')

            def create_dataloaders(self):
                loader = FakeLoader()
                return loader, loader, loader

            def train_one_epoch(self, *args):
                return 0.2

            def evaluate(self, model, data_loader, device, name='Evaluation'):
                return {
                    'target_loss': 0.15,
                    'identity_loss': 0.5,
                }

        with TemporaryDirectory() as temporary_directory:
            paths = ProjectPaths(
                build_dir=Path(temporary_directory),
            )
            config = CQNSGTTrainingConfig(num_epochs=2)
            trainer = FastTrainer(training_config=config, paths=paths)
            legacy_model = torch.nn.Linear(ACTION_VECTOR_SIZE, 1)
            legacy_path = paths.build_dir / 'legacy.pth'
            trainer.save_checkpoint(legacy_model, legacy_path)

            with patch(
                'resonance.training.cq_nsgt_trainer.CQNSGTTransitionModel',
                lambda action_vector_size: torch.nn.Linear(
                    action_vector_size,
                    1,
                ),
            ):
                trainer.run(resume_from=legacy_path, start_epoch=1)

            latest = torch.load(
                paths.build_dir / config.latest_checkpoint_name,
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(latest['completed_epochs'], 2)
            self.assertEqual(latest['best_val_loss'], 0.15)
            self.assertIn('optimizer_state_dict', latest)


if __name__ == '__main__':
    unittest.main()
