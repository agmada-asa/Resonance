import hashlib
import json
import math
from bisect import bisect_right
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from resonance.config import DEFAULT_CONFIG
from resonance.data.dataset import SyntheticCQNSGTDataset


CQ_NSGT_CACHE_VERSION = 1
DEFAULT_CQ_NSGT_CACHE_DIRNAME = 'vector_cache_v1'
DEFAULT_CQ_NSGT_SHARD_SIZE = 128
DEFAULT_CQ_NSGT_SPLITS = ('train', 'val', 'test')


# Store only the audio settings that change the generated CQ-NSGT vectors
def _audio_config_dict(config):
    return {
        'sample_rate': config.sample_rate,
        'duration': config.duration,
        'bins_per_octave': config.bins_per_octave,
        'fmin': config.fmin,
    }


# Hash the source metadata so stale vectors cannot be used after regeneration
def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# Replace the manifest only after the complete JSON has reached the filesystem
def _atomic_json_dump(value, path):
    path = Path(path)
    temporary_path = path.with_suffix(f'{path.suffix}.tmp')
    with temporary_path.open('w') as destination:
        json.dump(value, destination, indent=2)
        destination.write('\n')
    temporary_path.replace(path)


# Keep an interrupted save from replacing the last complete tensor shard
def _atomic_torch_save(value, path):
    path = Path(path)
    temporary_path = path.with_suffix(f'{path.suffix}.tmp')
    torch.save(value, temporary_path)
    temporary_path.replace(path)


# Describe every shard before generation so an interrupted run can be resumed
def _planned_shards(split, sample_count, shard_size):
    return [
        {
            'path': f'{split}/shard_{start // shard_size:05d}.pt',
            'start': start,
            'length': min(shard_size, sample_count - start),
        }
        for start in range(0, sample_count, shard_size)
    ]


# Check the inexpensive structural properties before a shard is trusted
def _validate_shard(payload, descriptor, split):
    expected = {
        'cache_version': CQ_NSGT_CACHE_VERSION,
        'split': split,
        'start': descriptor['start'],
    }
    for name, expected_value in expected.items():
        if payload.get(name) != expected_value:
            raise ValueError(
                f'Cached shard has incompatible {name}: '
                f'{payload.get(name)} != {expected_value}.'
            )

    action_vectors = payload.get('action_vectors')
    input_blocks = payload.get('input_blocks')
    target_blocks = payload.get('target_blocks')
    length = descriptor['length']
    if not isinstance(action_vectors, torch.Tensor):
        raise ValueError('Cached shard has no action vector tensor.')
    if action_vectors.shape[0] != length:
        raise ValueError('Cached shard has the wrong sample count.')
    if not isinstance(input_blocks, list) or not isinstance(target_blocks, list):
        raise ValueError('Cached shard blocks must be stored as lists.')
    if len(input_blocks) != len(target_blocks):
        raise ValueError('Cached input and target block counts do not match.')
    for input_block, target_block in zip(input_blocks, target_blocks):
        if input_block.shape != target_block.shape:
            raise ValueError('Cached input and target block shapes do not match.')
        if input_block.shape[0] != length:
            raise ValueError('Cached block has the wrong sample count.')


# Confirm representative samples remain bit-identical after serialization
def _verify_shard_samples(expected, actual):
    sample_count = expected['action_vectors'].shape[0]
    sample_indices = sorted({0, sample_count // 2, sample_count - 1})
    for index in sample_indices:
        torch.testing.assert_close(
            actual['action_vectors'][index],
            expected['action_vectors'][index],
            rtol=0,
            atol=0,
        )
        for actual_block, expected_block in zip(
            actual['input_blocks'],
            expected['input_blocks'],
        ):
            torch.testing.assert_close(
                actual_block[index],
                expected_block[index],
                rtol=0,
                atol=0,
            )
        for actual_block, expected_block in zip(
            actual['target_blocks'],
            expected['target_blocks'],
        ):
            torch.testing.assert_close(
                actual_block[index],
                expected_block[index],
                rtol=0,
                atol=0,
            )


# Stack matching ragged blocks across samples into one shard-level tensor list
def _collate_cache_samples(samples, split, start):
    if not samples:
        raise ValueError('Cannot write an empty CQ-NSGT cache shard.')

    block_count = len(samples[0]['input'])
    if any(len(sample['input']) != block_count for sample in samples):
        raise ValueError('CQ-NSGT input block counts are inconsistent.')
    if any(len(sample['target']) != block_count for sample in samples):
        raise ValueError('CQ-NSGT target block counts are inconsistent.')

    input_blocks = [
        torch.stack([sample['input'][index] for sample in samples])
        for index in range(block_count)
    ]
    target_blocks = [
        torch.stack([sample['target'][index] for sample in samples])
        for index in range(block_count)
    ]
    action_vectors = torch.stack([
        sample['action_vector']
        for sample in samples
    ])

    return {
        'cache_version': CQ_NSGT_CACHE_VERSION,
        'split': split,
        'start': start,
        'input_blocks': input_blocks,
        'target_blocks': target_blocks,
        'action_vectors': action_vectors,
    }


# Decide whether an existing partial cache belongs to this exact dataset
def _manifest_matches(manifest, config, shard_size, sources):
    return (
        manifest.get('cache_version') == CQ_NSGT_CACHE_VERSION
        and manifest.get('audio_config') == _audio_config_dict(config)
        and manifest.get('shard_size') == shard_size
        and manifest.get('sources') == sources
    )


# Validate all files before treating a previously completed cache as reusable
def _validate_manifest_shards(cache_dir, manifest):
    for split, split_manifest in manifest['splits'].items():
        for descriptor in split_manifest['shards']:
            shard_path = cache_dir / descriptor['path']
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f'CQ-NSGT cache shard does not exist: {shard_path}'
                )
            payload = torch.load(
                shard_path,
                map_location='cpu',
                weights_only=True,
                mmap=True,
            )
            _validate_shard(payload, descriptor, split)


def precompute_cq_nsgt_cache(
    data_dir,
    config=DEFAULT_CONFIG,
    cache_dirname=DEFAULT_CQ_NSGT_CACHE_DIRNAME,
    shard_size=DEFAULT_CQ_NSGT_SHARD_SIZE,
    splits=DEFAULT_CQ_NSGT_SPLITS,
    overwrite=False,
):
    if shard_size <= 0:
        raise ValueError('shard_size must be positive.')

    data_dir = Path(data_dir)
    cache_dir = data_dir / cache_dirname
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'manifest.json'

    # Keep each cached split tied to the metadata that recreates its audio
    metadata_paths = {
        split: data_dir / f'metadata_{split}.jsonl'
        for split in splits
    }
    missing = [
        str(path)
        for path in metadata_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            'Cannot precompute CQ-NSGT vectors; metadata is missing: '
            + ', '.join(missing)
        )

    # Record content hashes rather than relying on filenames or timestamps
    sources = {
        split: {
            'path': metadata_paths[split].name,
            'sha256': _sha256(metadata_paths[split]),
        }
        for split in splits
    }
    # Reuse a compatible cache or continue its unfinished shard sequence
    existing_manifest = None
    if manifest_path.is_file():
        with manifest_path.open() as source:
            existing_manifest = json.load(source)

    if existing_manifest and not overwrite:
        if not _manifest_matches(existing_manifest, config, shard_size, sources):
            raise ValueError(
                f'Existing cache at {cache_dir} was built with different '
                'metadata or settings. Re-run with --overwrite.'
            )
        if existing_manifest.get('complete'):
            _validate_manifest_shards(cache_dir, existing_manifest)
            print(f'CQ-NSGT vector cache is already complete: {cache_dir}')
            return existing_manifest

    # Count samples without constructing three large CQ-NSGT filter banks
    split_manifests = {}
    for split, metadata_path in metadata_paths.items():
        with metadata_path.open() as source:
            sample_count = sum(1 for _ in source)
        if sample_count == 0:
            raise ValueError(f'CQ-NSGT {split} metadata is empty.')
        split_manifests[split] = {
            'sample_count': sample_count,
            'shards': _planned_shards(split, sample_count, shard_size),
        }

    # Write an incomplete manifest first so interruption is always detectable
    manifest = {
        'cache_version': CQ_NSGT_CACHE_VERSION,
        'representation': 'cq_nsgt_complex_vector_shards',
        'complete': False,
        'audio_config': _audio_config_dict(config),
        'shard_size': shard_size,
        'sources': sources,
        'splits': split_manifests,
    }
    _atomic_json_dump(manifest, manifest_path)

    # Build one split at a time to release its analysis filter bank afterward
    for split, metadata_path in metadata_paths.items():
        dataset = SyntheticCQNSGTDataset(metadata_path, config=config)
        expected_count = split_manifests[split]['sample_count']
        if len(dataset) != expected_count:
            raise ValueError(
                f'{split} metadata changed while the cache was being built.'
            )
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # Overwrite removes only recognized shard files inside this cache split
        if overwrite:
            for old_shard in split_dir.glob('shard_*.pt'):
                old_shard.unlink()

        descriptors = split_manifests[split]['shards']
        for shard_number, descriptor in enumerate(descriptors, start=1):
            shard_path = cache_dir / descriptor['path']
            # A valid completed shard does not need to be transformed again
            if shard_path.is_file() and not overwrite:
                payload = torch.load(
                    shard_path,
                    map_location='cpu',
                    weights_only=True,
                    mmap=True,
                )
                _validate_shard(payload, descriptor, split)
                print(
                    f'Reusing {split} shard '
                    f'[{shard_number}/{len(descriptors)}]'
                )
                continue

            start = descriptor['start']
            stop = start + descriptor['length']
            # Recreate and transform only the samples assigned to this shard
            samples = []
            for index in range(start, stop):
                samples.append(dataset[index])
                processed = index - start + 1
                if processed % 16 == 0 or index + 1 == stop:
                    print(
                        f'  Precomputing {split} shard '
                        f'[{shard_number}/{len(descriptors)}] '
                        f'sample [{processed}/{descriptor["length"]}]',
                        flush=True,
                    )

            # Save atomically, then compare selected values with the source data
            payload = _collate_cache_samples(samples, split, start)
            _validate_shard(payload, descriptor, split)
            _atomic_torch_save(payload, shard_path)
            saved_payload = torch.load(
                shard_path,
                map_location='cpu',
                weights_only=True,
                mmap=True,
            )
            _validate_shard(saved_payload, descriptor, split)
            _verify_shard_samples(payload, saved_payload)
            print(
                f'Saved {split} shard [{shard_number}/{len(descriptors)}]: '
                f'{shard_path}',
                flush=True,
            )
        del dataset

    # Record the shared ragged layout for diagnostics and compatibility checks
    first_descriptor = next(
        descriptor
        for split_manifest in split_manifests.values()
        for descriptor in split_manifest['shards']
    )
    first_shard = torch.load(
        cache_dir / first_descriptor['path'],
        map_location='cpu',
        weights_only=True,
        mmap=True,
    )
    manifest['block_count'] = len(first_shard['input_blocks'])
    manifest['block_shapes'] = [
        list(block.shape[1:])
        for block in first_shard['input_blocks']
    ]
    manifest['action_vector_size'] = first_shard['action_vectors'].shape[1]
    manifest['complete'] = True
    _atomic_json_dump(manifest, manifest_path)
    print(f'CQ-NSGT vector cache complete: {cache_dir}')
    return manifest


class CachedCQNSGTDataset(Dataset):
    def __init__(self, cache_dir, split, config=DEFAULT_CONFIG):
        self.cache_dir = Path(cache_dir)
        self.split = split
        manifest_path = self.cache_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'CQ-NSGT cache manifest does not exist: {manifest_path}. '
                'Run experiments/precompute_cq_nsgt_vectors.py first.'
            )
        # Refuse incomplete, old, or differently configured vector caches
        with manifest_path.open() as source:
            self.manifest = json.load(source)
        if not self.manifest.get('complete'):
            raise RuntimeError(
                f'CQ-NSGT cache is incomplete: {self.cache_dir}. '
                'Resume the precompute command.'
            )
        if self.manifest.get('cache_version') != CQ_NSGT_CACHE_VERSION:
            raise ValueError('Unsupported CQ-NSGT cache version.')
        if self.manifest.get('audio_config') != _audio_config_dict(config):
            raise ValueError('CQ-NSGT cache audio settings do not match.')
        if split not in self.manifest['splits']:
            raise ValueError(f'CQ-NSGT cache has no {split!r} split.')
        # Detect metadata edits made after the vectors were precomputed
        source = self.manifest.get('sources', {}).get(split)
        if source is None:
            raise ValueError('CQ-NSGT cache has no source metadata record.')
        source_path = self.cache_dir.parent / source['path']
        if not source_path.is_file() or _sha256(source_path) != source['sha256']:
            raise ValueError(
                f'CQ-NSGT source metadata changed after caching: {source_path}. '
                'Re-run precomputation with --overwrite.'
            )

        # Keep compact lookup tables for mapping global indices into shards
        split_manifest = self.manifest['splits'][split]
        self.shards = split_manifest['shards']
        self.shard_offsets = [shard['start'] for shard in self.shards]
        self.shard_lengths = [shard['length'] for shard in self.shards]
        self.sample_count = split_manifest['sample_count']
        self._loaded_shard_index = None
        self._loaded_shard = None

    def __len__(self):
        return self.sample_count

    def _load_shard(self, shard_index):
        # Memory-map one shard at a time instead of loading the full 10 GB cache
        if self._loaded_shard_index != shard_index:
            descriptor = self.shards[shard_index]
            payload = torch.load(
                self.cache_dir / descriptor['path'],
                map_location='cpu',
                weights_only=True,
                mmap=True,
            )
            _validate_shard(payload, descriptor, self.split)
            self._loaded_shard_index = shard_index
            self._loaded_shard = payload
        return self._loaded_shard

    def __getitem__(self, index):
        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)

        # Convert the dataset index into a shard index and local sample index
        shard_index = bisect_right(self.shard_offsets, index) - 1
        descriptor = self.shards[shard_index]
        local_index = index - descriptor['start']
        payload = self._load_shard(shard_index)
        return {
            'input': [block[local_index] for block in payload['input_blocks']],
            'target': [block[local_index] for block in payload['target_blocks']],
            'action_vector': payload['action_vectors'][local_index],
        }


class CQNSGTShardBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
        seed=0,
    ):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        # Change the deterministic shuffle order between training epochs
        self.epoch = epoch

    def __iter__(self):
        # Shuffle whole shards first so nearby batches reuse the same mapping
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_indices = list(range(len(self.dataset.shards)))
        if self.shuffle:
            order = torch.randperm(
                len(shard_indices),
                generator=generator,
            ).tolist()
            shard_indices = [shard_indices[index] for index in order]

        for shard_index in shard_indices:
            descriptor = self.dataset.shards[shard_index]
            indices = list(range(
                descriptor['start'],
                descriptor['start'] + descriptor['length'],
            ))
            # Shuffle samples inside the shard without mixing shard boundaries
            if self.shuffle:
                order = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()
                indices = [indices[index] for index in order]

            # Shard-local batches prevent random access from thrashing the cache
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        if self.drop_last:
            return sum(
                length // self.batch_size
                for length in self.dataset.shard_lengths
            )
        return sum(
            math.ceil(length / self.batch_size)
            for length in self.dataset.shard_lengths
        )
