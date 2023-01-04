import numpy as np
from utils import *
from model import ChessModel
import tensorflow as tf
from constants import *
from datetime import datetime
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os


# BatchMCTS settings
num_sims_per_move: int = 1200
temperature: float = 1.0
autoplay: bool = True
output_directory = "./games"
output: str = output_directory + "/game"
num_threads: int = 8
batch_size: int = 4096
num_sectors: int = 2
cpuct: float = 0.01
boards: np.ndarray = np.zeros([num_sectors, batch_size, ROWS, COLS], dtype=np.int32)
boards_reshaped: np.ndarray = boards.reshape([num_sectors * batch_size, ROWS, COLS])
metadata: np.ndarray = np.zeros([num_sectors, batch_size, METADATA_LENGTH], dtype=np.int32)
metadata_reshaped: np.ndarray = metadata.reshape([num_sectors * batch_size, METADATA_LENGTH])
tablebase_path: str = "../backend/tablebase"
num_moves_per_inference = 250

# initialize
tablebase_path = c_char_p(bytes(tablebase_path, encoding="utf8"))
BatchMCTSExtension.initialize(tablebase_path)

# delete games
if output_directory:
    os.system("rm {0}/*".format(output_directory))

# play options
play_options = {
    "num_sims_per_move": 2000,
    "temperature": 1.0,
    "autoplay": False,
    "num_threads": 4,
    "batch_size": 200,
    "num_sectors": 1,
    "cpuct": cpuct,
}

# model settings
num_layers = 2
depth = 48
d_fnn = 60

# train settings
checkpoint_dir = "./checkpoints"
saved_model_dir = "./saved_model"
log_file = open("./log.txt", "a")
regularization_weight = 1e-4
train_batch_size = 4096

batch_mcts = BatchMCTS(
    num_sims_per_move,
    temperature,
    autoplay,
    output,
    num_threads,
    batch_size,
    num_sectors,
    cpuct,
    boards_reshaped,
    metadata_reshaped,
)

model = ChessModel(num_layers, depth, d_fnn)
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)
status = checkpoint.restore(manager.latest_checkpoint)
# compile the model
model(boards[0], metadata[0])
model.compile(optimizer=optimizer)
model.summary()

######################### Helper functions ###############################


def pnl(x):
    # print n log
    print(x)
    log_file.write(x + "\n")
    log_file.flush()


def update(out_q, out_policy):
    batch_mcts.update(out_q.reshape(-1), out_policy)


@tf.function
def inference_helper(b, m):
    result = model(b, m)
    out_policy_, out_q_ = result
    tf.numpy_function(func=update, inp=[out_q_, out_policy_], Tout=[])


def get_trtfunc(model):
    """
    def func(a, b):
        outp, outq = model(a, b)
        return {"output_1": outp, "output_2": outq}

    return func
    """
    model.save(saved_model_dir, save_format="tf")
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, precision_mode=trt.TrtPrecisionMode.FP32)
    trt_func = converter.convert()

    def input_fn():
        yield [boards[0], metadata[0]]

    converter.build(input_fn=input_fn)

    return trt_func


###################### Train loop! ###############################
# does inference using batch_mcts
def inference(loop_num):
    pnl("began inference loop {0}!".format(loop_num))
    # trt_func = get_trtfunc(model)
    for i in range(num_moves_per_inference * num_sims_per_move):
        batch_mcts.select()
        cur_sector = batch_mcts.current_sector()
        b = boards[cur_sector]
        m = metadata[cur_sector]
        inference_helper(b, m)
        if i > 0 and i % (100 * num_sims_per_move) == 0:
            pnl(
                "finished inference move {0}/{1} in loop {2}!".format(
                    i // num_sims_per_move, num_moves_per_inference, loop_num
                )
            )
        if i % num_sims_per_move == 0 and i:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("finished move", i // num_sims_per_move, "current time is:", current_time)
    pnl("finished inference loop {0}!".format(loop_num))


def train(loopidx):
    num_games = len(os.listdir(output_directory)) - batch_size * num_sectors
    pnl("Training on {0} games! Guessing {1} batches".format(num_games, num_games * 250 / train_batch_size))
    train_loss = tf.keras.metrics.Mean()
    for i, batch in enumerate(generate_batches_from_directory(output_directory, train_batch_size)):
        print("batch:", i)
        b = batch["board"]
        m = batch["metadata"]
        p = batch["policy"]
        p = p.reshape([p.shape[0], -1])  # (batch size, num_moves)
        v = batch["value"]
        l = batch["legal moves"]
        l = l.reshape([l.shape[0], -1])
        with tf.GradientTape() as tape:
            out_policy, out_value = model(b, m)
            out_value = tf.reshape(out_value, [-1])
            out_policy = tf.reshape(out_policy, [out_policy.shape[0], -1])  # (batch size, num_moves)
            out_policy += (1 - l) * -1e6  # illegal moves should be zeroed out
            out_policy = tf.nn.log_softmax(out_policy, axis=-1)
            loss = tf.pow(v - out_value, 2) - tf.reduce_sum(out_policy * p * l, axis=1)  # per batch loss
            train_loss.update_state(loss)  # update loss values
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_variables]) * regularization_weight
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    delete_finished_games(output_directory)
    pnl("finished train loop {0}! loss was {1}".format(loopidx, train_loss.result().numpy()))


index = 0
while True:
    inference(index)
    train(index)
    manager.save()
    if index and index % 10 == 0:
        pnl("playing games...")
        model2 = ChessModel(num_layers, depth, d_fnn)
        checkpoint2 = tf.train.Checkpoint(model=model2)
        manager2 = tf.train.CheckpointManager(checkpoint2, directory=checkpoint_dir, max_to_keep=10)
        status = checkpoint2.restore(manager2.checkpoints()[0])

        results = play(
            model,
            model2,
            play_options,
        )
        results.pop("results")  # dont wanna see this array...
        pnl("Game Results: {0}".format(results))
    index += 1
