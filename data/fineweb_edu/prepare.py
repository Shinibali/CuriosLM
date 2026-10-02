"""
Tokenize HuggingFaceFW/fineweb-edu (sample-10BT) into fixed-size shards. All logic is in data/sharding.py.

Run: python data/fineweb_edu/prepare.py
"""
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DATA_DIR))  # data/fineweb_edu -> data -> root
sys.path.insert(0, PROJECT_ROOT)  # this script lives outside the curioslm package, so the root
                                   # isn't on sys.path when it runs as a plain script
from curioslm.config import DataConfig  # noqa: E402 -- must follow the sys.path fix above
from data.sharding import prepare  # noqa: E402

_data_cfg = DataConfig()


def main():
    prepare(_data_cfg.dataset, DATA_DIR, "HuggingFaceFW/fineweb-edu", "sample-10BT", _data_cfg.shard_size,
            approx_total_tokens=10_000_000_000)  # FineWeb-Edu's dataset card figure for sample-10BT


if __name__ == "__main__":
    main()
