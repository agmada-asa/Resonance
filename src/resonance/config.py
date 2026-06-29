from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 44100  # Hz
    duration: int = 2  # seconds
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # Hop length for STFT
    n_mels: int = 128  # Number of Mel bands
    bins_per_octave: int = 12  # Number of bins per octave for CQT - use 36 for higher frequency resolution
    n_bins: int = 113  # Number of bins for CQT, covering C1 through roughly 20 kHz without exceeding Nyquist
    fmin: float = 32.70  # Minimum frequency for CQT (C1 note)
    cqt_db_floor: float = -80.0  # Fixed CQT decibel floor, not relative to each clip


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path = ROOT / "data/synthetic/v001"
    build_dir: Path = ROOT / "build"


DEFAULT_CONFIG = AudioConfig()
DEFAULT_PATHS = ProjectPaths()

SAMPLE_RATE = DEFAULT_CONFIG.sample_rate  # Hz
DURATION = DEFAULT_CONFIG.duration  # seconds
N_FFT = DEFAULT_CONFIG.n_fft  # FFT window size
HOP_LENGTH = DEFAULT_CONFIG.hop_length  # Hop length for STFT
N_MELS = DEFAULT_CONFIG.n_mels  # Number of Mel bands
BINS_PER_OCTAVE = DEFAULT_CONFIG.bins_per_octave  # Number of bins per octave for CQT - use 36 for higher frequency resolution
N_BINS = DEFAULT_CONFIG.n_bins  # Number of bins for CQT, covering C1 through roughly 20 kHz without exceeding Nyquist
FMIN = DEFAULT_CONFIG.fmin  # Minimum frequency for CQT (C1 note)
CQT_DB_FLOOR = DEFAULT_CONFIG.cqt_db_floor  # Fixed CQT decibel floor, not relative to each clip
