"""Checks for the streaming loader. Run from the project root: uv run python -m chess_engine.test_loader"""
import resource
import sys
import time

import numpy as np
import torch

from chess_engine.encoding import decode_position
from chess_engine.loader import _prepare_loader

SHARDS = ["data/Adams.npz", "data/Alekhine.npz", "data/Anand.npz", "data/Aronian.npz"]


def load_rows(shard):
    data = np.load(shard)
    return data["boards"], data["turns"], data["moves"]


def test_shapes_and_types():
    x, y_from, y_to = next(_prepare_loader(SHARDS))
    assert x.shape == (256, 769) and x.dtype == torch.float32, (x.shape, x.dtype)
    assert y_from.shape == (256,) and y_from.dtype == torch.int64, (y_from.shape, y_from.dtype)
    assert y_to.shape == (256,) and y_to.dtype == torch.int64, (y_to.shape, y_to.dtype)
    print("shapes and types: ok")


def test_rows_match_shard():
    # Every (board, turn, move) handed out must be a real row of the shard, i.e. nothing got misaligned.
    boards, turns, moves = load_rows(SHARDS[0])
    real_rows = set(zip(map(bytes, boards), turns.tolist(), map(bytes, moves)))

    checked = 0
    for x, y_from, y_to in _prepare_loader([SHARDS[0]]):
        for k in range(len(y_from)):
            row, turn = decode_position(x[k].numpy())
            move = np.array([y_from[k], y_to[k]], dtype=np.uint8)
            assert (bytes(row), turn, bytes(move)) in real_rows, f"batch row {k} is not a real shard row"
            checked += 1
        if checked >= 2560:
            break
    print(f"rows match shard: ok ({checked} rows checked)")


def test_every_row_once_per_epoch():
    expected = sum(len(load_rows(shard)[0]) for shard in SHARDS)
    seen = sum(len(y_from) for _, y_from, _ in _prepare_loader(SHARDS))
    assert seen == expected, f"saw {seen} rows, shards have {expected}"
    print(f"every row once per epoch: ok ({seen} rows)")


def test_reproducible_per_epoch():
    first = lambda epoch: next(_prepare_loader(SHARDS, seed=42, epoch=epoch))[0]
    assert torch.equal(first(0), first(0)), "same seed and epoch gave different batches"
    assert not torch.equal(first(0), first(1)), "different epochs gave the same batch"
    print("reproducible per epoch: ok")


def report_throughput_and_memory():
    # A full epoch over several shards, so it crosses shard boundaries.
    # ru_maxrss is the peak memory so far (bytes on macOS, KB on Linux); flat means no growth per shard.
    to_mb = 1 / 2**20 if sys.platform == "darwin" else 1 / 2**10
    start = time.perf_counter()
    n_batches = 0
    for n_batches, _ in enumerate(_prepare_loader(SHARDS), start=1):
        if n_batches % 400 == 0:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * to_mb
            print(f"  batch {n_batches}: peak memory {peak:.0f} MB")
    elapsed = time.perf_counter() - start
    print(f"throughput: {n_batches / elapsed:.1f} batches/sec ({n_batches * 256 / elapsed:,.0f} positions/sec)")


if __name__ == "__main__":
    test_shapes_and_types()
    test_rows_match_shard()
    test_every_row_once_per_epoch()
    test_reproducible_per_epoch()
    report_throughput_and_memory()
