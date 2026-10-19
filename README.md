# chess-engine

A small chess engine, made for fun while learning ML. Alpha-beta search, with a small neural network (trained on master games) that decides which moves to search first.

Training: change the settings in `src/chess_engine/train.py`, then run `uv run python -m chess_engine.train`.
Training data goes in `data/` (built from PGNs with `src/chess_engine/chess_data.py`), trained weights in `models/`.

Coming soon: a Lichess bot.
