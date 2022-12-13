import numpy as np
from utils import *

# from model import ChessModel

# import tensorflow as tf

num_sims_per_move: int = 1600
temperature: float = 1.0
autoplay: bool = True
output: str = "./games/bruh"
num_threads: int = 4
batch_size: int = 512
num_sectors: int = 1
cpuct: float = 0.1
boards: np.ndarray = np.zeros([num_sectors, batch_size, ROWS, COLS]).astype(int)
metadata: np.ndarray = np.zeros([num_sectors, batch_size, METADATA_LENGTH]).astype(int)

batch_mcts = BatchMCTS(
    num_sims_per_move,
    temperature,
    autoplay,
    output,
    num_threads,
    batch_size,
    num_sectors,
    cpuct,
    boards.reshape([-1, ROWS, COLS]),
    metadata.reshape([-1, METADATA_LENGTH]),
)
print()
print(batch_mcts.proportion_of_games_over())
print(batch_mcts.current_sector())
print(batch_mcts.all_games_over())
