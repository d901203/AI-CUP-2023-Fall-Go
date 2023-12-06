from collections import defaultdict

import numpy as np


def iter_neighbors(board, point):
    row, col = point
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = row + dx, col + dy
        if 0 <= x < board.side and 0 <= y < board.side:
            yield x, y


def get_board_string(board):
    point_to_color = {p: c for c, p in board.list_occupied_points()}
    visited = {}
    liberties = defaultdict(set)
    liberties_neighboring = defaultdict(int)
    i = -1
    for point in point_to_color.keys():
        if point in visited:
            continue
        i += 1
        stack = [point]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited[cur] = i
            for nxt in iter_neighbors(board, cur):
                if nxt not in point_to_color:
                    liberties[i].add(nxt)
                    liberties_neighboring[nxt] += 1
                else:
                    if nxt in visited:
                        assert (point_to_color[cur] != point_to_color[nxt]) or visited[cur] == visited[nxt]
                        continue
                    if point_to_color[cur] == point_to_color[nxt]:
                        stack.append(nxt)
    return visited, liberties, liberties_neighboring


def get_board_info(board, friend_color):
    empty = np.zeros((board.side, board.side), dtype=np.bool_)
    friend = np.zeros((board.side, board.side), dtype=np.bool_)
    enemy = np.zeros((board.side, board.side), dtype=np.bool_)
    liberties = np.zeros((board.side, board.side), dtype=np.int_)
    visited, board_liberties, _ = get_board_string(board)
    for row in range(board.side):
        for col in range(board.side):
            color = board.get(row, col)
            idx = visited.get((row, col), None)
            if color is None:
                empty[row, col] = True
            elif color == friend_color:
                friend[row, col] = True
                if idx is not None:
                    liberties[row, col] = len(board_liberties[idx])
            else:
                enemy[row, col] = True
                if idx is not None:
                    liberties[row, col] = len(board_liberties[idx])
    return empty, friend, enemy, liberties


def get_history_planes(board_side, history):
    result = np.zeros((board_side, board_side))
    for time, (row, col) in enumerate(history, start=1):
        result[row, col] = time
    return len(history) + 1 - result


class Encoder:
    def __init__(self):
        self.num_planes = 17

    def encode(self, game_state, predict_color, history, ko=None):
        result = np.zeros((self.num_planes, game_state.side, game_state.side), dtype=np.bool_)
        empty, friend, enemy, liberties = get_board_info(game_state, predict_color)

        result[0] = friend
        result[1] = enemy
        result[2] = empty
        result[3] = 1

        result[4] = friend & (liberties == 1)
        result[5] = friend & (liberties == 2)
        result[6] = friend & (liberties == 3)
        result[7] = friend & (liberties >= 4)

        result[8] = enemy & (liberties == 1)
        result[9] = enemy & (liberties == 2)
        result[10] = enemy & (liberties == 3)
        result[11] = enemy & (liberties >= 4)

        history_planes = get_history_planes(game_state.side, history)

        result[12] = history_planes == 1
        result[13] = history_planes == 2
        result[14] = history_planes == 3
        result[15] = history_planes == 4

        if ko is not None:
            result[16][ko[0]][ko[1]] = 1

        return result
