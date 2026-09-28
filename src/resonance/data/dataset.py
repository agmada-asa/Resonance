import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from resonance.actions import AudioActionProcessor
from resonance.config import (
    ACTION_DIMENSIONS,
    ACTION_VECTOR_SIZE,
    MAX_ACTIONS,
    DEFAULT_CONFIG,
)
from resonance.data.synthetic import WaveformSynthesizer
from resonance.features.cq_nsgt import CQNSGTVectorCodec


class SyntheticSpectrogramDataset(Dataset):
    def __init__(self, npz_path, mean, std, device=None):
        npz_path = Path(npz_path)

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
        
        if device is not None:
            self.inputs = self.inputs.to(device)
            self.target_deltas = self.target_deltas.to(device)
            self.targets = self.targets.to(device)
            self.action_vectors = self.action_vectors.to(device)

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


class SyntheticCQNSGTDataset(Dataset):
    def __init__(self, metadata_path, config=DEFAULT_CONFIG):
        metadata_path = Path(metadata_path)
        self.config = config

        # Load the instructions required to recreate every synthetic example
        with metadata_path.open() as f:
            self.metadata = [json.loads(line) for line in f]

        # Stop old metadata being used with a different audio configuration
        if self.metadata:
            metadata_config = self.metadata[0]
            expected_settings = {
                'sample_rate': config.sample_rate,
                'duration': config.duration,
                'bins_per_octave': config.bins_per_octave,
                'fmin': config.fmin,
            }
            for name, expected_value in expected_settings.items():
                if metadata_config.get(name) != expected_value:
                    raise ValueError(
                        f'CQ-NSGT metadata {name} does not match the '
                        f'current config: {metadata_config.get(name)} != '
                        f'{expected_value}.'
                    )

        self.action_processor = AudioActionProcessor(config)
        self.waveform_synthesizer = WaveformSynthesizer(config)
        self.codec = CQNSGTVectorCodec(config=config)

    def _generate_input_audio(self, metadata):
        # Recreate either the original waveform or the original three-note chord
        generate_audio = (
            self.waveform_synthesizer.generate_chords
            if metadata.get('is_chord', False)
            else self.waveform_synthesizer.generate_waveform
        )
        return generate_audio(
            metadata['waveform_type'],
            metadata['frequency'],
            metadata['amplitude'],
        )

    def _apply_actions(self, audio, metadata):
        target_audio = audio
        encoded_actions = np.zeros(
            (MAX_ACTIONS, ACTION_DIMENSIONS),
            dtype=np.float32,
        )

        # Apply every action in order and place its encoding in the same slot
        for slot, action in enumerate(metadata.get('actions', [])):
            target_audio, action_vector = self.action_processor.apply_action(
                target_audio,
                action['action'],
                action['parameter'],
            )
            encoded_actions[slot] = action_vector

        action_vector = encoded_actions.reshape(ACTION_VECTOR_SIZE)
        return target_audio, torch.from_numpy(action_vector)

    @staticmethod
    def _remove_codec_batch(vector_blocks):
        # The codec analyses one item as a batch of one. Remove that dimension
        # so PyTorch's DataLoader can add the real training batch dimension.
        return [vectors.squeeze(0) for vectors in vector_blocks]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        input_audio = self._generate_input_audio(metadata)
        target_audio, action_vector = self._apply_actions(input_audio, metadata)

        # Convert both audio states into matching CQ-NSGT vector block lists
        input_blocks = self._remove_codec_batch(
            self.codec.audio_to_vectors(input_audio)
        )
        target_blocks = self._remove_codec_batch(
            self.codec.audio_to_vectors(target_audio)
        )
        return {
            'input': input_blocks,
            'target': target_blocks,
            'action_vector': action_vector,
        }
