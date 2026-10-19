import torch
import torch.nn.functional as F
import chess
from chess_engine.model import ChessNN
from chess_engine.chess_data import board_to_array
from chess_engine.encoding import encode_position


# Keep the loaded model in a module-level variable so it's loaded only once, not on every call.
# The engine runs on CPU: for one position at a time, copying to a GPU costs more than it saves.
# Shape must match the checkpoint: ChessNN() is the big model (1024 x 3); the old small ones need ChessNN(hidden=512, layers=2).
model = ChessNN()
model.load_state_dict(torch.load("models/final_model.pt", map_location="cpu"))
model.eval()


def board_to_tensor(board: chess.Board):
    state_board = board_to_array(board)
    enc_board = encode_position(state_board, board.turn)
    x_board = torch.tensor(enc_board, dtype=torch.float32).reshape(shape=(1, 769))
    return x_board


def model_scores(board: chess.Board):
    with torch.no_grad():
        from_logits, to_logits = model(board_to_tensor(board))

    return from_logits, to_logits


def model_log_probs(board: chess.Board):
    """Log-probabilities over the 64 squares for each head, shape (64,) each."""
    from_logits, to_logits = model_scores(board)
    # dim=1 is the 64 squares (dim=0 is the batch of one); [0] then drops the batch row.
    return F.log_softmax(from_logits, dim=1)[0], F.log_softmax(to_logits, dim=1)[0]


def move_score(move: chess.Move, from_list, to_list):
    """log P(from) + log P(to): the model's log-probability of this move."""
    return from_list[move.from_square] + to_list[move.to_square]


# Own copy of the piece values: importing them from minimax_ab would be a circular import.
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def capture_score(board: chess.Board, move: chess.Move):
    """MVV-LVA: most valuable victim first, least valuable attacker as tie-break (PxQ = 89, QxP = 1)."""
    if board.is_en_passant(move):
        victim = chess.PAWN  # the captured pawn is beside the to-square, which is empty
    else:
        victim = board.piece_at(move.to_square).piece_type
    attacker = board.piece_at(move.from_square).piece_type
    return PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker]


def order_moves(board: chess.Board):
    """Legal moves of this position, best first: captures by MVV-LVA, then quiet moves by the model."""
    # One model call for the whole position; plain lists are much faster to index than tensors.
    from_lp, to_lp = model_log_probs(board)
    from_list, to_list = from_lp.tolist(), to_lp.tolist()

    # Tuples compare element by element: group 1 (captures) sorts before group 0 (quiet moves),
    # and the second element orders the moves inside each group.
    def key(move):
        if board.is_capture(move):
            return (1, capture_score(board, move))
        return (0, move_score(move, from_list, to_list))

    return sorted(board.legal_moves, key=key, reverse=True)


def _print_top(board: chess.Board, n=5):
    from_lp, to_lp = model_log_probs(board)
    from_list, to_list = from_lp.tolist(), to_lp.tolist()
    ranked = order_moves(board)
    assert set(ranked) == set(board.legal_moves) and len(ranked) == board.legal_moves.count()
    print(f"{len(ranked)} legal moves, top {n}:")
    for move in ranked[:n]:
        if board.is_capture(move):
            print(f"  {move.uci()} capture, MVV-LVA {capture_score(board, move)}")
        else:
            print(f"  {move.uci()} model {move_score(move, from_list, to_list):.2f}")


if __name__ == "__main__":
    print(sum(p.numel() for p in model.parameters()))
    print(model.training)
    board = chess.Board()
    print(board_to_tensor(board).sum().item())
    board.push_san("e4")
    print(board_to_tensor(board).sum().item())

    board = chess.Board()
    from_logits, to_logits = model_scores(board)

    print(from_logits.shape, to_logits.shape)
    from_logits_max = torch.argmax(from_logits, dim=1)
    to_logits_max = torch.argmax(to_logits, dim=1)
    print(from_logits_max, to_logits_max)
    print(int(from_logits_max.float().mean().item()), int(to_logits_max.float().mean().item()))

    from_lp, to_lp = model_log_probs(board)
    print(from_lp.shape, to_lp.shape)
    print("max values (must be <= 0):", from_lp.max().item(), to_lp.max().item())
    print("probabilities sum to:", from_lp.exp().sum().item(), to_lp.exp().sum().item())
    top_from, top_to = from_lp.argmax().item(), to_lp.argmax().item()
    print("top squares:", chess.square_name(top_from), chess.square_name(top_to))
    print(f"P(from {chess.square_name(top_from)}) = {from_lp[top_from].exp().item():.2f}, "
          f"P(to {chess.square_name(top_to)}) = {to_lp[top_to].exp().item():.2f}")

    print("\n----------------------------")
    _print_top(chess.Board())
    # Black queen on h4 hangs to the knight on f3.
    _print_top(chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"))
    _print_top(chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"))
    # En passant: e5xd6 must be scored as a pawn capture, not crash on the empty d6.
    _print_top(chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"))
