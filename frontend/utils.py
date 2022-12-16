import numpy as np
from constants import *
import os
from collections import defaultdict, deque
from numpyctypes import c_ndarray
from numpy.ctypeslib import load_library
from ctypes import *

BatchMCTSExtension = load_library("extension_BatchMCTS", "../backend/output")
BatchMCTSExtension.initialize.argtypes = [POINTER(c_char)]
BatchMCTSExtension.deleteBatchMCTS.argtypes = [POINTER(c_char)]
BatchMCTSExtension.createBatchMCTS.argtypes = [
    c_int,
    c_float,
    c_bool,
    POINTER(c_char),
    c_int,
    c_int,
    c_int,
    c_float,
    Structure,
    Structure,
]
BatchMCTSExtension.select.argtypes = [POINTER(c_char)]
BatchMCTSExtension.update.argtypes = [POINTER(c_char), Structure, Structure]
BatchMCTSExtension.set_temperature.argtypes = [POINTER(c_char), c_float]
BatchMCTSExtension.play_best_moves.argtypes = [POINTER(c_char), c_bool]
BatchMCTSExtension.all_games_over.argtypes = [POINTER(c_char)]
BatchMCTSExtension.proportion_of_games_over.argtypes = [POINTER(c_char)]
BatchMCTSExtension.results.argtypes = [POINTER(c_char), Structure]
BatchMCTSExtension.current_sector.argtypes = [POINTER(c_char)]

BatchMCTSExtension.createBatchMCTS.restype = POINTER(c_char)
BatchMCTSExtension.all_games_over.restype = c_bool
BatchMCTSExtension.proportion_of_games_over.restype = c_double
BatchMCTSExtension.current_sector.restype = c_int


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
        boards_: np.ndarray,
        metadata_: np.ndarray,
        tablebase_path: str,
    ) -> None:
        boards = c_ndarray(boards_)
        metadata = c_ndarray(metadata_)
        # make caches to keep arrays in memory as required by BatchMCTS
        self.select_cache = [boards, metadata, boards_, metadata_]
        self.update_cache = deque()
        self.num_sectors = num_sectors
        output = c_char_p(bytes(output, encoding="utf8"))
        tablebase_path = c_char_p(bytes(tablebase_path, encoding="utf8"))
        BatchMCTSExtension.initialize(tablebase_path)
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

    def update(self, q_: np.ndarray, policy_: np.ndarray) -> None:
        q = c_ndarray(q_)
        policy = c_ndarray(policy_)
        self.update_cache.append((q, policy, q_, policy_))
        if len(self.update_cache) > self.num_sectors * 4:
            self.update_cache.popleft()
        BatchMCTSExtension.update(self.ptr, q, policy)

    def set_temperature(self, temp: float) -> None:
        BatchMCTSExtension.set_temperature(self.ptr, c_float(temp))

    def play_best_moves(self, reset: bool) -> None:
        BatchMCTSExtension.play_best_moves(self.ptr, c_bool(reset))

    def all_games_over(self) -> bool:
        return BatchMCTSExtension.all_games_over(self.ptr)

    def proportion_of_games_over(self) -> float:
        return BatchMCTSExtension.proportion_of_games_over(self.ptr)

    def results(self, res: np.ndarray) -> None:
        res = c_ndarray(res)
        BatchMCTSExtension.results(self.ptr, res)

    def current_sector(self) -> int:
        return BatchMCTSExtension.current_sector(self.ptr)


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
        assert abs(1 - np.sum(policy)) < 0.001
        yield {
            "board": board.reshape(ROWS, COLS),
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
        return {
            "board": board,
            "metadata": metadata,
            "policy": policy,
            "value": value,
            "legal moves": legal_moves,
        }

    for example in generate_examples_from_directory(dir):
        res.append(example)
        if len(res) == batch_size:
            yield result()
            res = []

    if len(res) > 0:
        yield result()


# plays 2 * num_games games given both models
# requires: batchmcts has auto-play off, num_sectors = 1
# batchmctsoptions: all args except for boards, metadata
def play(model1, model2, batchmctsoptions: dict):
    assert not batchmctsoptions["autoplay"]
    assert batchmctsoptions["num_sectors"] == 1
    batch_size = batchmctsoptions["batch_size"]
    assert batch_size % 2 == 0
    split = batch_size // 2
    num_sims_per_move = batchmctsoptions["num_sims_per_move"]
    boards: np.ndarray = np.zeros([batch_size, ROWS, COLS], dtype=np.int32)
    metadata: np.ndarray = np.zeros([batch_size, METADATA_LENGTH], dtype=np.int32)

    batchmctsoptions["boards_"] = boards
    batchmctsoptions["metadata_"] = metadata
    batchmcts = BatchMCTS(**batchmctsoptions)

    movenum = 1
    while not batchmcts.all_games_over():
        iswhite = movenum % 2
        for _ in range(num_sims_per_move):
            batchmcts.select()
            b1 = boards[:split]
            m1 = metadata[:split]
            b2 = boards[split:]
            m2 = metadata[split:]
            out_policy1, out_q1 = model1.call(b1, m1) if iswhite else model2.call(b1, m1)
            out_policy2, out_q2 = model2.call(b2, m2) if iswhite else model1.call(b2, m2)
            out_policy = np.concatenate([out_policy1.numpy(), out_policy2.numpy()], axis=0).astype(np.float32)
            out_q = np.concatenate([out_q1.numpy(), out_q2.numpy()], axis=0).flatten().astype(np.float32)
            batchmcts.update(out_q, out_policy)
        batchmcts.play_best_moves(reset=True)
        print("finished playing move number {0}".format(movenum))
        print("proportion of games over: {0}".format(batchmcts.proportion_of_games_over()))
        movenum += 1
    results = np.zeros([batch_size], dtype=np.int32)
    batchmcts.results(results)
    results[split:] *= -1  # since model1 plays as black on results[split:]
    scores = {}
    for i, r in enumerate([results[:split], results[split:]]):
        unique, counts = np.unique(r, return_counts=True)
        scores["black" if i else "white"] = dict(zip(unique, counts))
    scores["results"] = results
    return scores
