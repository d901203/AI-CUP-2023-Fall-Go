import numpy as np
import torch
from sgfmill import boards, sgf
from tqdm.auto import tqdm
from utils import Encoder

chars = "abcdefghijklmnopqrs"
char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = {i: c for i, c in enumerate(chars)}

encoder = Encoder()


def initialize_board():
    return boards.Board(19)


def process_moves(move, board):
    color, move_tuple = move.get_move()
    if color is not None and move_tuple is not None:
        row, col = move_tuple
        if board.get(row, col) is not None:
            board.apply_setup([], [], empty_points=[(row, col)])
        ko = board.play(row, col, color)
    return board, ko


def process_label(move):
    column = char_to_num[move[2]]
    row = char_to_num[move[3]]
    return column * 19 + row


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
        predict_color = "b" if len(history) % 2 == 0 else "w"
        X.append(encoder.encode(board, predict_color, history, ko))
    return np.array(X, dtype=np.float32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("./csv/dan_test_private.csv", "r") as f:
        df = f.readlines()

    games_id = [i.split(",", 2)[0] for i in df]
    games = [i.split(",", 2)[-1] for i in df]

    X = process_games(games)
    np.save("./data/dan/test/private", X)
