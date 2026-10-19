import chess
import time
from chess_engine.move_ordering import order_moves


count = 0

def bump_counter():
    global count
    count += 1


piece_value = {
    1:1,
    2:3,
    3:3,
    4:5,
    5:9,
    6:0
}


def evaluate(board: chess.Board) -> float:
    wsum, bsum = 0, 0
    for square, piece in board.piece_map().items():
        if piece.color:
            wsum += piece_value[piece.piece_type]
        else:
            bsum += piece_value[piece.piece_type]

    return wsum - bsum


MATE: int = 2_000_000

def minimax(board: chess.Board, depth: int, alpha, beta, maximizing: bool, use_model: bool = False) -> float:
    bump_counter()
    if board.is_checkmate():
        return MATE if not board.turn else -MATE
    if board.is_game_over():
        return 0
    if depth == 0:
        return evaluate(board)

    best = -float("inf") if maximizing else float("inf")

    loop_param = board.legal_moves if not use_model else order_moves(board)
    for move in loop_param:
        board.push(move)
        score = minimax(board, depth - 1, alpha, beta, not maximizing, use_model)
        board.pop()

        if maximizing:
            best = max(best, score)
            alpha = max(alpha, best)
        else:
            best = min(best, score)
            beta = min(beta, best)

        if alpha>=beta:
            break
    
    return best


def choose_move(board: chess.Board, depth=3, use_model: bool = False) -> tuple[float, chess.Move]:
    global count
    count = 0
    maximizing: bool = board.turn == chess.WHITE
    best_move = None
    best_score = -float("inf") if maximizing else float("inf")
    alpha, beta = -float("inf"), float("inf")

    loop_param = board.legal_moves if not use_model else order_moves(board)
    for move in loop_param:
        board.push(move)
        score = minimax(board, depth-1, alpha, beta, not maximizing, use_model)
        board.pop()

        if (maximizing and score > best_score) or (not maximizing and score < best_score):
            best_move, best_score = move, score

        if maximizing:
            alpha = max(alpha, best_score)
        else:
            beta = min(beta, best_score)

    return (best_score, best_move)


if __name__ == "__main__":
    # board = chess.Board()
    # print(choose_move(board), count, board)

    # board.push_san("e4")
    # print(choose_move(board), count, board)

    # print("\nUse model on\n----------------------")
    # board = chess.Board()
    # print(choose_move(board, use_model=True), count, board)

    # board.push_san("e4")
    # print(choose_move(board, use_model=True), count, board)


    boards: list = [
        {"name":"start", "fen":chess.STARTING_FEN},
        {"name":"ruy lopez", "fen":"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"},
        {"name":"scholar mate", "fen":"r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"},
        {"name":"hanging queen", "fen":"rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"},
        {"name":"kiwipete", "fen":"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"}
    ]


    # Same position, same depth, both modes: the score must match, only nodes and time may differ.
    for depth in (3, 4):
        for b in boards:
            results = {}
            for use_model in (False, True):
                board = chess.Board(fen=b["fen"])
                start_time = time.perf_counter()
                score, move = choose_move(board, depth=depth, use_model=use_model)
                results[use_model] = (score, move, count, time.perf_counter() - start_time)

            (s_off, m_off, n_off, t_off), (s_on, m_on, n_on, t_on) = results[False], results[True]
            print(f"depth {depth} | {b['name']:<13} | off: {s_off:>7} {m_off} {n_off:>7} nodes {t_off:.3f}s"
                  f" | on: {s_on:>7} {m_on} {n_on:>7} nodes {t_on:.3f}s | same score: {s_off == s_on}")
