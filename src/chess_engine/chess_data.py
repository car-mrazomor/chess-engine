import argparse
import hashlib
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.pgn
import numpy as np


def board_to_array(board: chess.Board) -> np.ndarray:
    """64 squares a1..h8: white pieces 1..6, black pieces -1..-6, empty 0."""
    state = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        state[square] = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
    return state


def game_hash(game: chess.pgn.Game) -> bytes:
    """Identifies a game by its start position and moves, used to drop duplicates across files."""
    text = game.board().fen() + " " + " ".join(m.uci() for m in game.mainline_moves())
    return hashlib.blake2b(text.encode(), digest_size=8).digest()


def pgn_to_arrays(pgn_path: str | Path) -> dict[str, np.ndarray]:
    """Parse every game in a PGN file into one row per position (the move played from it)."""
    boards, moves, promotions, turns, castling, ep_squares, plies, game_ids = [], [], [], [], [], [], [], []
    hashes = []

    with open(pgn_path, encoding="utf-8", errors="replace") as pgn_file:
        while game := chess.pgn.read_game(pgn_file):
            game_id = len(hashes)
            board = game.board()
            ply = 1

            for move in game.mainline_moves():
                boards.append(board_to_array(board))
                moves.append((move.from_square, move.to_square))
                promotions.append(move.promotion or 0)
                turns.append(board.turn == chess.WHITE)
                castling.append((
                    board.has_kingside_castling_rights(chess.WHITE),
                    board.has_queenside_castling_rights(chess.WHITE),
                    board.has_kingside_castling_rights(chess.BLACK),
                    board.has_queenside_castling_rights(chess.BLACK),
                ))
                ep_squares.append(board.ep_square if board.has_legal_en_passant() else -1)
                plies.append(ply)
                game_ids.append(game_id)

                ply += 1
                board.push(move)

            hashes.append(game_hash(game))

    return {
        "boards": np.array(boards, dtype=np.int8).reshape(-1, 64),
        "moves": np.array(moves, dtype=np.uint8).reshape(-1, 2),
        "promotions": np.array(promotions, dtype=np.int8),
        "turns": np.array(turns, dtype=np.bool_),
        "castling": np.array(castling, dtype=np.bool_).reshape(-1, 4),
        "ep_squares": np.array(ep_squares, dtype=np.int8),
        "plies": np.array(plies, dtype=np.int16),
        "game_ids": np.array(game_ids, dtype=np.int32),
        "game_hashes": np.array(hashes, dtype="S8"),
    }


def _parse(pgn_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    return pgn_path, pgn_to_arrays(pgn_path)


def build_dataset(pgn_dir: str | Path, out_dir: str | Path, workers: int | None = None) -> None:
    """Convert every PGN in pgn_dir into one compressed .npz shard per file, skipping duplicate games."""
    pgn_dir, out_dir = Path(pgn_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sorted so the same game is always kept in the same shard, regardless of worker order.
    pgn_paths = sorted(pgn_dir.glob("*.pgn"))
    seen: set[bytes] = set()
    total_games = total_positions = duplicates = 0

    with Pool(workers) as pool:
        for i, (pgn_path, data) in enumerate(pool.imap(_parse, pgn_paths), start=1):
            keep_game = np.zeros(len(data["game_hashes"]), dtype=np.bool_)
            for game_id, h in enumerate(data["game_hashes"]):
                if h not in seen:
                    seen.add(h)
                    keep_game[game_id] = True
            duplicates += int((~keep_game).sum())

            # Drop duplicate games, then renumber game_ids to stay contiguous within the shard.
            keep_row = keep_game[data["game_ids"]]
            new_ids = np.cumsum(keep_game) - 1
            shard = {k: v[keep_row] for k, v in data.items() if k != "game_hashes"}
            shard["game_ids"] = new_ids[shard["game_ids"]].astype(np.int32)

            if len(shard["boards"]):
                np.savez_compressed(out_dir / f"{pgn_path.stem}.npz", **shard)
            total_games += int(keep_game.sum())
            total_positions += len(shard["boards"])
            print(f"[{i}/{len(pgn_paths)}] {pgn_path.name}: {keep_game.sum()} games, "
                  f"{len(shard['boards'])} positions")

    print(f"Done: {total_games} games, {total_positions} positions, {duplicates} duplicate games skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PGN files into .npz training data.")
    parser.add_argument("pgn_dir", nargs="?", default="PGNs")
    parser.add_argument("out_dir", nargs="?", default="data")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    build_dataset(args.pgn_dir, args.out_dir, args.workers)
