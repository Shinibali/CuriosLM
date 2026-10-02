"""
Tokenize a ~5B-token slice of Cosmopedia-v2 (HuggingFaceTB/smollm-corpus) into fixed-size shards, for mixing
with FineWeb-Edu. All logic is in data/sharding.py; this file only sets what differs:
  - 51 shards (1 val + 50 train): a 50:50 mix over ~9.83B tokens needs ~4.92B of this.
  - A seeded shuffle first: the corpus is 104 parquet files and we take ~18% of it, so the slice shouldn't be
    whatever the first files happen to contain.

Run: python data/cosmopedia_v2/prepare.py
"""
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
sys.path.insert(0, PROJECT_ROOT)
from curioslm.config import DataConfig  # noqa: E402
from data.sharding import prepare  # noqa: E402


def main():
    # shard size must match FineWeb-Edu's: the loader assumes one shard size
    prepare("cosmopedia_v2", DATA_DIR, "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", DataConfig().shard_size,
            max_shards=51, shuffle_seed=1337)


if __name__ == "__main__":
    main()
    # Every shard and the checkpoint are on disk by now. Skip interpreter shutdown: abandoning a shuffled HF stream
    # mid-corpus leaves parquet reads in flight, and pyarrow's thread pool then waits on them forever at exit.
    sys.stdout.flush()
    os._exit(0)
