import numpy as np
import torch
from sgfmill import boards, sgf
from tqdm.auto import tqdm
from utils import Encoder

encoder = Encoder()


def initialize_board():
    return boards.Board(19)


def process_moves(move, board):
    color, move_tuple = move.get_move()
    ko_ = None
    if color is not None and move_tuple is not None:
        row, col = move_tuple
        if board.get(row, col) is not None:
            board.apply_setup([], [], empty_points=[(row, col)])
        ko_ = board.play(row, col, color)
    return board, ko_


def process_games(games):
    X = []
    for game in tqdm(games):
        board = initialize_board()
        sgf_game = sgf.Sgf_game.from_string("(;" + game.replace(",", ";") + ")")
        game = game.split(",")
        moves = sgf_game.get_main_sequence()
        history = []
        ko = None
        for move in moves:
            board, ko = process_moves(move, board)
            history.append(move.get_move()[1])
        X.append(encoder.encode(board, game[-1][0].lower(), history, ko))
    return np.array(X, dtype=np.bool_)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("./csv/play_style_test_public.csv", "r") as f:
        df = f.readlines()

    games_id = [i.split(",", 2)[0] for i in df]
    games = [i.split(",", 1)[-1] for i in df]
    cleaned_games = []
    for game in games:
        moves = [move for move in game.split(",") if move != "" and move.strip()]
        cleaned_game = ",".join(moves)
        cleaned_games.append(cleaned_game)

    X = process_games(cleaned_games)
    np.save("./data/play_style/test/public", X)
