import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]

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
