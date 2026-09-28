from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.data.synthetic import (  # noqa: E402
    SyntheticCQNSGTDataGenerator,
)


def main():
    generator = SyntheticCQNSGTDataGenerator()
    generator.generate_training_data(seed=42)


if __name__ == '__main__':
    main()
