import numpy as np
import random
from pathlib import Path


DATA_PATH = "data"


def _read_shard(shard):
    array = np.load(shard)
    boards = array["boards"]
    array.close()
    return len(boards)


def _running_total(shards):
    return sum([_read_shard(shard) for shard in shards])


def _split():
    temp_array = []
    for p in Path(DATA_PATH).rglob("*"):
        if not p.name.endswith(".npz"):
            continue
        if p in temp_array:
            continue
        temp_array.append(p)

    temp_array = sorted(temp_array)
    random.seed(42)
    random.shuffle(temp_array)

    val_shards = temp_array[:int(len(temp_array)*0.10)]
    train_shards = temp_array[int(len(temp_array)*0.10):]

    return (
        val_shards, train_shards, _running_total(val_shards), _running_total(train_shards), temp_array
    )


def _check():
    f1 = _split()
    f2 = _split()
    assert f1[0] == f2[0], "reproducibility"
    assert f1[1] == f2[1], "reproducibility"
    assert f1[0] == f2[0] and f2[1] == f1[1], "reproducibility"
    assert not (set(f1[0]) & set(f1[1])) and not (set(f2[0]) & set(f2[1])), "Overlapping"
    assert len(f1[0]) + len(f1[1]) == len(f1[4]), "Something lost"


if __name__ == "__main__":
    _check()