import io
import chess
import chess.pgn
from chess_engine.minimax_ab import choose_move


def check_turn(moves: str, turn: str) -> bool:
    board = chess.Board()
    for move in moves.split(" "):
        board.push_uci(move)
    if turn == "white" and board.turn:
        return True
    if turn == "black" and not board.turn:
        return True
    
    return False


def check_turn_from_fen(fen: str) -> bool:
    board = chess.Board(fen)
    return board.turn


def build_board(moves: str = None, first_to_play: bool = False):
    if first_to_play:
        return chess.Board()

    board = chess.Board()
    game = chess.pgn.read_game(io.StringIO(moves))
    for move in game.mainline_moves():
        board.push(move)

    return board


def _get_move(moves: str = None):
    board: chess.Board = build_board(moves)
    score, move = choose_move(board, depth=3, use_model=True)

    return move


def _first_to_play() -> str:
    board = build_board(first_to_play=True)
    score, move = choose_move(board, depth=3, use_model=True)

    return move
