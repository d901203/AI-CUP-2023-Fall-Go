import os
from multiprocessing import Pool

import numpy as np
from sgfmill import boards, sgf
from tqdm.auto import tqdm
from utils import Encoder

NUM_CORES = 30  # 使用的執行緒數量
BATCH_SIZE = 4096  # 每次處理的資料數量
CSV_PATH = "./csv/play_style_train.csv"  # 訓練集 CSV 路徑
DATA_DIR = "./data/play_style"  # 訓練檔案存放資料夾


def flip_horizontal(board):
    return np.flip(board, axis=2)


def flip_vertical(board):
    return np.flip(board, axis=1)


def rotate_90(board):
    return np.rot90(board, k=1, axes=(1, 2))


def rotate_180(board):
    return np.rot90(board, k=2, axes=(1, 2))


def rotate_270(board):
    return np.rot90(board, k=3, axes=(1, 2))


def flip_diagonal(board):
    return np.transpose(board, axes=(0, 2, 1))


def flip_antidiagonal(board):
    return np.flip(flip_diagonal(board), axis=2)


def flip(X):
    return [
        X,
        flip_horizontal(X),
        flip_vertical(X),
        rotate_90(X),
        rotate_180(X),
        rotate_270(X),
        flip_diagonal(X),
        flip_antidiagonal(X),
    ]


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


def process_games(sub_games):
    X = []
    y = []
    for game in sub_games:
        label, game = game
        board = initialize_board()
        sgf_game = sgf.Sgf_game.from_string("(;" + game.replace(",", ";") + ")")
        game = game.split(",")
        moves = sgf_game.get_main_sequence()
        history = []
        for move in moves:
            board, ko = process_moves(move, board)
            history.append(move.get_move()[1])
        for symm in flip(encoder.encode(board, game[-1][0].lower(), history, ko)):
            X.append(symm)
            y.append(label)
    return np.array(X, dtype=np.bool_), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    with open(CSV_PATH, "r") as f:
        df = f.readlines()

    games = []
    for row in df:
        row = row.split(",", 2)
        games.append((int(row[1]) - 1, row[-1].strip()))

    sub_lists = np.array_split(games, NUM_CORES)

    all_x = []
    all_y = []

    for i, sub_lists in enumerate(np.array_split(games, len(games) // BATCH_SIZE)):
        sub_list = np.array_split(sub_lists, NUM_CORES)
        with Pool(NUM_CORES) as p:
            results = list(tqdm(p.imap(process_games, sub_list), total=len(sub_list), desc=f"Processing Batch {i+1}"))

        x = np.concatenate([result[0] for result in results])
        y = np.concatenate([result[1] for result in results])

        all_x.append(x)
        all_y.append(y)

    with open(f"{DATA_DIR}/train_x.npz", "wb") as f:
        np.save(f, np.concatenate(all_x))

    with open(f"{DATA_DIR}/train_y.npz", "wb") as f:
        np.save(f, np.concatenate(all_y))
