import torch
import numpy as np
from chess_engine.encoding import encode_position


def _split_batches(boards, turns, moves, size):
    for i in range(0, len(boards), size):
        rows = np.array([encode_position(r, turns[i:i+size][_]) for _,r in enumerate(boards[i:i+size])], dtype=np.float32)
        x = torch.tensor(rows, dtype=torch.float32)
        _from = np.array([f[0] for f in moves[i:i+size]], dtype=np.int64)
        y_from = torch.tensor(_from, dtype=torch.int64)
        _to = np.array([t[1] for t in moves[i:i+size]], dtype=np.int64)
        y_to = torch.tensor(_to, dtype=torch.int64)
        yield (x, y_from, y_to)


def _prepare_loader(shards, seed=42, epoch=0):
    # Seeding with both values: every epoch gets a new order, and each epoch is reproducible.
    rng = np.random.default_rng([seed, epoch])
    shards = rng.permutation(shards)
    for t_shard in shards:
        data_array = np.load(t_shard)
        boards = data_array["boards"]
        moves = data_array["moves"]
        turns = data_array["turns"]

        order = rng.permutation(len(boards))

        boards = boards[order]
        moves = moves[order]
        turns = turns[order]

        for x_batch, y_from_batch, y_to_batch in _split_batches(boards, turns, moves, 256):
            yield (x_batch, y_from_batch, y_to_batch)
