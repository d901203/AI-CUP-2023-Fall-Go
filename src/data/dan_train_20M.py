import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from sgfmill import boards, sgf
from tqdm.auto import tqdm
from utils import Encoder

NUM_CORES = 30  # 使用的執行緒數量
BATCH_SIZE = 16384  # 每次處理的資料數量
CSV_PATH = "./csv/dan_train.csv"  # 訓練集 CSV 路徑
DATA_DIR = "./data/dan/20M"  # 訓練檔案存放資料夾

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


def process_games(sub_games):
    X = []
    y = []
    for game in sub_games:
        color, game = game
        board = initialize_board()
        sgf_game = sgf.Sgf_game.from_string("(;" + game.replace(",", ";") + ")")
        game = game.split(",")
        moves = sgf_game.get_main_sequence()
        history = []
        ko = None
        for i, move in enumerate(moves[:-1]):
            board, ko = process_moves(move, board)
            history.append((move.get_move()[1]))
            predict_color = game[i + 1][0].lower()
            if color.lower() == predict_color:
                continue
            X.append(encoder.encode(board, predict_color, history, ko))
            y.append(process_label(game[i + 1]))
    return np.array(X, dtype=np.bool_), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    with open(CSV_PATH, "r") as f:
        df = f.readlines()

    games = []
    for row in df:
        row = row.split(",", 2)
        games.append((row[1], row[-1].strip()))

    for i, sub_lists in enumerate(np.array_split(games, len(games) // BATCH_SIZE)):
        sub_list = np.array_split(sub_lists, NUM_CORES)
        process = partial(process_games)
        with Pool(NUM_CORES) as p:
            results = list(
                tqdm(
                    p.imap(process, sub_list),
                    total=len(sub_list),
                    desc=f"Processing Batch {i+1} for {BATCH_SIZE} games",
                )
            )

        x = np.concatenate([result[0] for result in results])
        y = np.concatenate([result[1] for result in results])

        with open(f"{DATA_DIR}/train_x_{i + 1}.npz", "wb") as f:
            np.save(f, x)

        with open(f"{DATA_DIR}/train_y_{i + 1}.npz", "wb") as f:
            np.save(f, y)
