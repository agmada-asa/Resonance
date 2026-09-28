import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.config import DEFAULT_PATHS  # noqa: E402
from resonance.data.cq_nsgt_cache import (  # noqa: E402
    DEFAULT_CQ_NSGT_CACHE_DIRNAME,
    DEFAULT_CQ_NSGT_SHARD_SIZE,
    precompute_cq_nsgt_cache,
)


# Parse cache sizing and recovery options without changing project defaults
def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute sharded CQ-NSGT vectors for training.',
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=DEFAULT_CQ_NSGT_SHARD_SIZE,
        help='samples per shard (default: %(default)s)',
    )
    parser.add_argument(
        '--cache-dirname',
        default=DEFAULT_CQ_NSGT_CACHE_DIRNAME,
        help='cache directory created below the CQ-NSGT data directory',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='replace an existing cache instead of resuming it',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Generate every split into the configured versioned cache directory
    precompute_cq_nsgt_cache(
        DEFAULT_PATHS.cq_nsgt_data_dir,
        cache_dirname=args.cache_dirname,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
