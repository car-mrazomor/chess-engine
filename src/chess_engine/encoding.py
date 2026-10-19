import numpy as np
import chess


def encode_position(row: np.ndarray, turn: bool):
    piece_values = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
    planes = row[None, :] == piece_values[:, None]
    planes = np.append(planes, turn)
    return planes.astype(np.float32).flatten()


def decode_position(array: np.ndarray):
    array = array.astype(np.int8)
    turn: bool = True if array[-1:][0] == 1 else False
    piece_values = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
    new_array = [0 for i in range(64)]
    for i, arr in enumerate(np.reshape(array[:-1], (12, 64))):
        for _, a in enumerate(arr):
            if a == 1:
                new_array[_] = piece_values[i]

    return np.array(new_array, dtype=np.int8), turn


def encode_move(move: tuple[int, int]):
    return move


def decode_move(board: chess.Board, move: tuple[int, int], turn: bool):
    piece = board.piece_type_at(int(move[0]))
    piece = piece if turn else -piece
    if (not turn and int(move[1]) in range(0,8) and piece == -1) or \
        (turn and int(move[1]) in range(64-8,64) and piece == 1):
        return chess.Move(int(move[0]), int(move[1]), promotion=chess.QUEEN)

    return chess.Move(int(move[0]), int(move[1]))


def round_trip(board_row, move, turn, castling, ep_square) -> bool:
    encoded_row = encode_position(board_row, turn)
    decoded_row, decoded_turn = decode_position(encoded_row)

    if not np.array_equal(decoded_row, board_row) or decoded_turn != bool(turn):
        return False

    # Build the board from the DECODED data, so the test also checks decode_position.
    board = chess.Board.empty()
    board.turn = decoded_turn

    for i, p in enumerate(decoded_row):
        if p != 0:
            piece = chess.Piece(abs(int(p)), chess.WHITE if p > 0 else chess.BLACK)
            board.set_piece_at(i, piece)

    # 64 piece values have no history, so restore what the game history knew.
    # castling flags: white kingside, white queenside, black kingside, black queenside
    rook_squares = [chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]
    board.castling_rights = 0
    for has_right, rook_square in zip(castling, rook_squares):
        if has_right:
            board.castling_rights |= rook_square

    if ep_square >= 0:
        board.ep_square = int(ep_square)

    encoded_move = encode_move(move)
    decoded_move = decode_move(board, encoded_move, decoded_turn)

    return decoded_move in board.legal_moves


if __name__ == "__main__":
    data_array = np.load("data/Adams.npz")
    # Load each array once; indexing data_array["..."] decompresses it every time.
    boards = data_array["boards"]
    moves = data_array["moves"]
    turns = data_array["turns"]
    castlings = data_array["castling"]
    ep_squares = data_array["ep_squares"]
    promotions = data_array["promotions"]

    def count_fails(rows):
        return sum(
            not round_trip(boards[i], moves[i], turns[i], castlings[i], ep_squares[i])
            for i in rows
        )

    rng = np.random.default_rng()
    random_rows = rng.choice(len(boards), size=1000, replace=False)
    promotion_rows = np.nonzero(promotions > 0)[0]
    # En passant capture: a pawn moving onto the en passant square.
    en_passant_rows = [
        i for i in np.nonzero(ep_squares >= 0)[0]
        if moves[i][1] == ep_squares[i] and abs(boards[i][moves[i][0]]) == chess.PAWN
    ]

    for name, rows in [("random", random_rows), ("promotions", promotion_rows), ("en passant", en_passant_rows)]:
        print(f"{name}: {len(rows)} rows, {count_fails(rows)} fails")
