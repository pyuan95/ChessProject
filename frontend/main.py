import numpy as np
from utils import *
from model import ChessModel
import tensorflow as tf
from collections import deque
from constants import *

# BatchMCTS settings
num_sims_per_move: int = 1600
temperature: float = 1.0
autoplay: bool = True
output_directory = "./games"
output: str = output_directory + "/game"
output = ""  # for now...
num_threads: int = 4
batch_size: int = 16
num_sectors: int = 1
cpuct: float = 0.05
boards: np.ndarray = np.zeros([num_sectors, batch_size, ROWS, COLS]).astype(int)
metadata: np.ndarray = np.zeros([num_sectors, batch_size, METADATA_LENGTH]).astype(int)

# model settings
num_layers = 4
depth = 64
d_fnn = 96

# train settings
num_moves_per_inference = 1000
checkpoint_dir = "./checkpoints"
log_file = open("./log.txt", "a")
regularization_weight = 1e-4

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

model = ChessModel(num_layers, depth, d_fnn)
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)
status = checkpoint.restore(manager.latest_checkpoint)
cache = deque()

##############################################
# Train loop!


def p(x):
    print(x)
    log_file.write(x + "\n")
    log_file.flush()


# does inference using batch_mcts
def inference(loop_num):
    for i in range(num_moves_per_inference):
        batch_mcts.select()
        cur_sector = batch_mcts.current_sector()
        b = boards[cur_sector]
        m = metadata[cur_sector]
        policy, q = model.call(b, m)
        policy = policy.numpy()
        q = q.numpy().flatten()
        cache.append((policy, q))
        if len(cache) > num_sectors * 4:
            cache.popleft()

        batch_mcts.update(q, policy)

        if i > 0 and i % 250 == 0:
            p("finished inference sim {0} in loop {1}!".format(i, loop_num))
    p("finished inference loop {0}!".format(loop_num))


def train(loopidx):
    num_games = len(os.listdir(output_directory)) - batch_size * num_sectors
    p("Training on {0} games!".format(num_games))
    train_loss = tf.keras.metrics.Mean()
    for i, batch in enumerate(generate_batches_from_directory(output_directory, batch_size)):
        b = batch["board"]
        m = batch["metadata"]
        p = batch["policy"]
        p = p.reshape([p.shape[0], -1])  # (batch size, num_moves)
        v = batch["value"]
        l = batch["legal moves"]
        with tf.GradientTape() as tape:
            out_policy, out_value = model.call(b, m)
            out_value = out_value.reshape([-1])
            out_policy += (1 - l) * -1e9  # illegal moves should be zeroed out
            out_policy = tf.reshape(out_policy, [out_policy.shape[0], -1])  # (batch size, num_moves)
            out_policy = tf.nn.softmax(out_policy, axis=-1)
            loss = tf.pow(v - out_value, 2) - tf.reduce_sum(tf.math.log(out_policy) * p, axis=1)  # per batch loss
            train_loss.update_state(loss)  # update loss values
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_variables]) * regularization_weight
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    p("finished train loop {0}! loss was {1}".format(loopidx, train_loss.result().numpy()))


index = 0
while True:
    inference(index)
    train(index)
    manager.save()
    index += 1


# inference
# batch_mcts.select()
