import numpy as np
from constants import *
import os
from collections import defaultdict
from numpyctypes import c_ndarray
from numpy.ctypeslib import load_library
from ctypes import *

BatchMCTSExtension = load_library("extension_BatchMCTS", "../backend/output")
BatchMCTSExtension.proportion_of_games_over.restype = c_float
BatchMCTSExtension.all_games_over.restype = c_bool
BatchMCTSExtension.current_sector.restype = c_int


def generate_examples(lines):
    i = 0
    value = lines[-1].split(" ")[0]
    value = int(value)
    for i in range(0, len(lines) - 1, 4):
        l1 = np.array([int(x) for x in lines[i].split(",")[:-1]])
        assert len(l1) == ROWS * COLS + METADATA_LENGTH
        board = l1[: ROWS * COLS]
        metadata = l1[ROWS * COLS :]
        policy = np.zeros([ROWS, COLS, NUM_MOVES_PER_SQUARE])
        legal_moves = np.zeros([ROWS, COLS, NUM_MOVES_PER_SQUARE])
        moves = lines[i + 1].split(",")[:-1]
        color = lines[i + 3].strip("\n")
        for j in range(0, len(moves), 4):
            r = int(moves[j])
            c = int(moves[j + 1])
            i = int(moves[j + 2])
            p = float(moves[j + 3])
            policy[r, c, i] = p
            legal_moves[r, c, i] = 1
        yield {
            "board": board,
            "metadata": metadata,
            "policy": policy,
            "value": value if color == "WHITE" else value * -1,
            "legal moves": legal_moves,
        }


def get_finished_games(dir):
    def get_prefix_suffix(file_name):
        x = file_name.split("_")
        return tuple(x[:-1]), int(x[-1])

    files = os.listdir(dir)
    prefix_to_largest_suffix = defaultdict(lambda: -1)
    for f in files:
        prefix, suffix = get_prefix_suffix(f)
        prefix_to_largest_suffix[prefix] = max(prefix_to_largest_suffix[prefix], suffix)

    def filter(f):
        prefix, suffix = get_prefix_suffix(f)
        return prefix_to_largest_suffix[prefix] > suffix

    return [f for f in files if filter(f)]


def delete_finished_games(dir):
    for f in get_finished_games(dir):
        os.remove(os.path.join(dir, f))


def generate_examples_from_directory(dir):
    for f in get_finished_games(dir):
        lines = open(os.path.join(dir, f)).readlines()
        for example in generate_examples(lines):
            yield example


def generate_batches_from_directory(dir, batch_size):
    res = []

    def result():
        board = np.concatenate([x["board"][None, ...] for x in res], axis=0)
        metadata = np.concatenate([x["metadata"][None, ...] for x in res], axis=0)
        policy = np.concatenate([x["policy"][None, ...] for x in res], axis=0)
        legal_moves = np.concatenate([x["legal moves"][None, ...] for x in res], axis=0)
        value = np.array([x["value"] for x in res])
        return {"board": board, "metadata": metadata, "policy": policy, "value": value, "legal moves": legal_moves}

    for example in generate_examples_from_directory(dir):
        res.append(example)
        if len(res) == batch_size:
            yield result()
            res = []

    if len(res) > 0:
        yield result()


class BatchMCTS:
    def __init__(
        self,
        num_sims_per_move: int,
        temperature: float,
        autoplay: bool,
        output: str,
        num_threads: int,
        batch_size: int,
        num_sectors: int,
        cpuct: float,
        boards: np.ndarray,
        metadata: np.ndarray,
        tablebase_path: str,
    ) -> None:
        boards = c_ndarray(boards, dtype=int, ndim=len(boards.shape), shape=boards.shape)
        metadata = c_ndarray(metadata, dtype=int, ndim=len(metadata.shape), shape=metadata.shape)
        output = c_char_p(bytes(output, encoding="utf8"))
        tablebase_path = c_char_p(bytes(tablebase_path, encoding="utf8"))
        BatchMCTSExtension.intiialize(tablebase_path)
        self.ptr = BatchMCTSExtension.createBatchMCTS(
            num_sims_per_move,
            c_float(temperature),
            autoplay,
            output,
            num_threads,
            batch_size,
            num_sectors,
            c_float(cpuct),
            boards,
            metadata,
        )

    def cleanup(self) -> None:
        BatchMCTSExtension.deleteBatchMCTS(self.ptr)

    def select(self) -> None:
        BatchMCTSExtension.select(self.ptr)

    def update(self, q: np.ndarray, policy: np.ndarray) -> None:
        q = c_ndarray(q, dtype=np.float, ndim=len(q.shape), shape=q.shape)
        policy = c_ndarray(policy, dtype=np.float, ndim=len(policy.shape), shape=policy.shape)
        BatchMCTSExtension.update(self.ptr, q, policy)

    def set_temperature(self, temp: float) -> None:
        BatchMCTSExtension.set_temperature(self.ptr, c_float(temp))

    def play_best_moves(self) -> None:
        BatchMCTSExtension.play_best_moves(self.ptr)

    def all_games_over(self) -> bool:
        return BatchMCTSExtension.all_games_over(self.ptr)

    def proportion_of_games_over(self) -> float:
        return BatchMCTSExtension.proportion_of_games_over(self.ptr)

    def results(self, res: np.ndarray) -> None:
        res = c_ndarray(res, dtype=int, ndim=len(res.shape), shape=res.shape)
        BatchMCTSExtension.results(self.ptr, res)

    def current_sector(self) -> int:
        return BatchMCTSExtension.current_sector(self.ptr)
