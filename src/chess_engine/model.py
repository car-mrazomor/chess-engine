import time
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from chess_engine.loader import _prepare_loader
from chess_engine.pipeline import _split


# First match wins: NVIDIA GPU, then Mac GPU, else CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class ChessNN(nn.Module):
    # hidden/layers must match the checkpoint being loaded: the old models are hidden=512, layers=2.
    def __init__(self, hidden=1024, layers=3):
        super().__init__()

        # Named fc1, fc2, ... so the saved weights keep the same keys as before.
        self.n_layers = layers
        for i in range(1, layers + 1):
            setattr(self, f"fc{i}", nn.Linear(769 if i == 1 else hidden, hidden))
        self.from_head = nn.Linear(hidden, 64)
        self.to_head = nn.Linear(hidden, 64)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, f"fc{i}")(x))

        return self.from_head(x), self.to_head(x)


def _model():
    # Move before creating the optimizer: it keeps references to the model's parameters.
    model = ChessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loss_fn


def accuracy(from_logits, to_logits, y_from, y_to):
    from_guess = torch.argmax(from_logits, dim=1)   # shape (256,)
    to_guess = torch.argmax(to_logits, dim=1)       # shape (256,)

    from_right = from_guess == y_from
    to_right = to_guess == y_to
    both_right = from_right & to_right

    return (
        from_right.float().mean().item(),
        to_right.float().mean().item(),
        both_right.float().mean().item(),
    )


def _validate(model, loss_fn, val_shards):
    """Measure the model on data it never trains on. No backward pass, no optimizer step."""
    model.eval()
    total_loss = 0.0
    batches = rows = 0
    from_right = to_right = both_right = 0

    with torch.no_grad():
        for x_batch, y_from, y_to in _prepare_loader(val_shards):
            x_batch, y_from, y_to = x_batch.to(device), y_from.to(device), y_to.to(device)
            from_logits, to_logits = model(x_batch)
            total_loss += (loss_fn(from_logits, y_from) + loss_fn(to_logits, y_to)).item()
            batches += 1

            # accuracy() gives fractions; turn them back into counts so the smaller last batch isn't overweighted.
            n = len(y_from)
            from_acc, to_acc, both_acc = accuracy(from_logits, to_logits, y_from, y_to)
            from_right += round(from_acc * n)
            to_right += round(to_acc * n)
            both_right += round(both_acc * n)
            rows += n

    return total_loss / batches, from_right / rows, to_right / rows, both_right / rows


def _train(epochs=3, n_train_shards=10, checkpoint_dir="models"):
    val_shards, train_shards, lv, lt, ta = _split()
    model, optimizer, loss_fn = _model()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    settings = (f"run {time.strftime('%Y-%m-%d %H:%M')} | device {device} | {n_params:,} params | "
                f"{n_train_shards} train shards | {epochs} epochs | -> {checkpoint_dir}")
    print(settings)
    with open("logs/train_log.txt", "a") as f:
        f.write(f"\n{settings}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        start = time.perf_counter()

        train_loader = _prepare_loader(train_shards[:n_train_shards], epoch=epoch)

        for x_batch, y_from, y_to in train_loader:
            x_batch, y_from, y_to = x_batch.to(device), y_from.to(device), y_to.to(device)
            optimizer.zero_grad()
            from_logits, to_logits = model(x_batch)
            loss = loss_fn(from_logits, y_from) + loss_fn(to_logits, y_to)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            if batches % 1000 == 0:
                print(f"  epoch {epoch + 1}, batch {batches}: avg train loss {running_loss / batches:.3f}")

        train_loss = running_loss / batches
        val_loss, from_acc, to_acc, both_acc = _validate(model, loss_fn, val_shards)
        minutes, seconds = divmod(int(time.perf_counter() - start), 60)
        print(f"epoch {epoch + 1} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | "
              f"from {from_acc:.1%} | to {to_acc:.1%} | both {both_acc:.1%} | {minutes}m{seconds:02d}s")


        os.makedirs("logs", exist_ok=True)
        with open("logs/train_log.txt", "a") as f:
            f.write(f"epoch {epoch + 1} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | "
              f"from {from_acc:.1%} | to {to_acc:.1%} | both {both_acc:.1%} | {minutes}m{seconds:02d}s\n")

        checkpoint = Path(checkpoint_dir) / f"epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint)
        print(f"saved -> {checkpoint}")


def _checkpoint_test(checkpoint="models/epoch_3.pt"):
    """A saved model loaded into a fresh ChessNN must give the same val numbers as when it was trained."""
    val_shards, train_shards, lv, lt, ta = _split()
    model, optimizer, loss_fn = _model()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    val_loss, from_acc, to_acc, both_acc = _validate(model, loss_fn, val_shards)
    print(f"{checkpoint} | val loss {val_loss:.3f} | from {from_acc:.1%} | to {to_acc:.1%} | both {both_acc:.1%}")


def _overfit_test(shard="data/Adams.npz", steps=300):
    """Sanity check (4b): the model must be able to memorise a single batch."""
    model, optimizer, loss_fn = _model()
    model.train()

    x_batch, y_from, y_to = next(_prepare_loader([shard]))
    x_batch, y_from, y_to = x_batch.to(device), y_from.to(device), y_to.to(device)

    for step in range(steps):
        optimizer.zero_grad()
        from_logits, to_logits = model(x_batch)
        loss = loss_fn(from_logits, y_from) + loss_fn(to_logits, y_to)

        # Step 0 is before any update, so it shows the starting loss (expect ~8.3).
        if step % 50 == 0 or step == steps - 1:
            from_acc, to_acc, both_acc = accuracy(from_logits, to_logits, y_from, y_to)
            print(f"step {step}: loss {loss.item():.3f} | from {from_acc:.1%} | to {to_acc:.1%} | both {both_acc:.1%}")

        loss.backward()
        optimizer.step()
