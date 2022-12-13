import tensorflow as tf
from constants import *


def scaled_dot_product_attention(
    keys: tf.Tensor, queries: tf.Tensor, values: tf.Tensor
):
    """
    performs scaled dot product attention
    :param keys: the keys. shape: (batch size, sequence length, depth)
    :param queries: shape: (batch size, sequence length, depth)
    :param values: shape: (batch size, sequence length, depth)
    :return: values weighted by attention. shape: (batch size, sequence length, depth)
    """
    keys = tf.transpose(keys, [0, 2, 1])
    attention = tf.matmul(
        queries, keys
    )  # (batch size, sequence length q, sequence length k)
    attention = attention / tf.sqrt(
        tf.cast(tf.shape(keys)[2], dtype=tf.float32)
    )  # scaling
    attention = tf.keras.activations.softmax(attention, axis=-1)
    result = tf.matmul(attention, values)  # (batch size, sequence length q, depth)
    return result


def layer_norm(t: tf.Tensor):
    """
    Normalizes the vector representing each sequence element in each batch to N(0, 1)
    :param t: the input tensor. shape: (batch size, sequence length, depth)
    :return: the normalized tensor
    """
    mu = tf.reduce_mean(t, axis=2)  # e[x] shape=(batch size, seq_len)
    sigma = tf.reduce_mean(t * t, axis=2)  # e[x^2] shape=(batch size, seq_len)
    sigma = tf.sqrt(sigma - mu * mu)  # std dev of x
    return (t - mu[..., 1]) / sigma[..., 1]


class ChessModelLayer(tf.keras.layers.Layer):
    def __init__(self, depth: int, d_ffn: int):
        """
        Constructs a layer of the chess model
        :param depth: depth of the layer
        :param d_ffn: depth of the feed forward network
        """
        self.keys = tf.keras.layers.Dense(depth)
        self.queries = tf.keras.layers.Dense(depth)
        self.values = tf.keras.layers.Dense(depth)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_ffn, activation="relu"),
                tf.keras.layers.Dense(depth, activation="relu"),
            ]
        )

    def call(self, inp):
        """
        one layer of attention
        :param inp: the input. has shape: (batch size, sequence length, depth)
        :return: tensor with shape (batch size, sequence length, depth)
        """
        k = self.keys(inp)
        q = self.queries(inp)
        v = self.values(inp)
        result = scaled_dot_product_attention(k, q, v)
        result = self.ffn(result)
        result = layer_norm(result)
        return result


class ChessModel(tf.keras.Model):
    def __init__(self, num_layers, depth, d_fnn):
        super(ChessModel, self).__init__()
        rows_depth = int(depth // 2)
        cols_depth = depth - rows_depth
        self.depth = depth
        self.rows = tf.Variable(
            tf.random_normal_initializer()(
                shape=[1, ROWS, 1, rows_depth], dtype=tf.float32
            ),
            trainable=True,
        )
        self.cols = tf.Variable(
            tf.random_normal_initializer()(
                shape=[1, 1, COLS, cols_depth], dtype=tf.float32
            ),
            trainable=True,
        )
        self.positional_encoding = tf.concat(
            [
                self.rows + tf.zeros([1, ROWS, COLS, rows_depth], dtype=tf.float32),
                self.cols + tf.zeros([1, ROWS, COLS, cols_depth], dtype=tf.float32),
            ],
            axis=-1,
        )
        self.value_encoding = tf.Variable(
            tf.random_normal_initializer()(shape=[1, 1, depth], dtype=tf.float32),
            trainable=True,
        )
        self.policy = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(depth, activation="relu"),
                tf.keras.layers.Dense(NUM_MOVES_PER_SQUARE, activation="linear")
            ]
        )
        self.q = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(depth, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.encoding = tf.keras.layers.Dense(depth, use_bias=True)
        self.chess_layers = [ChessModelLayer(depth, d_fnn) for _ in range(num_layers)]

    def call(self, board, metadata):
        """
        calls the evaluator
        :param board: the chess board. shape: (batch size, ROWS, COLS), type = int
        :param metadata: the metadata shape: (batch size, METADATA), type = int
        :return: a tuple: (policy, value)
            shape of policy: (batch size, ROWS, COLS, NUM_MOVES_PER_SQUARE)
            shape of value: (batch size)
        """
        batch_size = tf.shape(board)[0]
        board = tf.one_hot(
            board, MAX_PIECE_ENCODING, dtype=tf.int32
        )  # (batch size, ROWS, COLS, MAX PIECE ENCODING)

        # the last element is the epsq
        # 64 represents no square
        epsq = metadata[:, -1]
        epsq = tf.one_hot(epsq, ROWS * COLS + 1, dtype=tf.int32)[:, :-1]  # (batch size, ROWS * COLS)
        epsq = tf.reshape(epsq, [-1, ROWS, COLS, 1])  # (batch size, ROWS, COLS, 1)
        metadata = metadata[:, None, None, :-1]  # (batch_size, 1, 1, METADATA - 1)
        metadata += tf.zeros(
            [1, ROWS, COLS, 1], dtype=tf.int32
        )  # broadcast to (batch size, ROWS, COLS, METADATA - 1)
        board = tf.concat(
            [board, metadata, epsq], axis=3
        )  # (batch size, ROWS, COLS, depth_0), type = int
        board = tf.cast(board, tf.float32)
        board = self.encoding(board)  # (batch size, ROWS, COLS, depth)
        board += self.positional_encoding
        board = tf.reshape(
            board, [-1, ROWS * COLS, self.depth]
        )  # (batch size, 64, depth)
        board = tf.concat(
            [
                board,
                self.value_encoding + tf.zeros([batch_size, 1, 1], dtype=tf.float32),
            ],
            axis=1,
        )  # (batch size, 65, depth)
        for layer in self.chess_layers:
            board = layer.call(board)
        q = board[:, ROWS * COLS, :]  # (batch_size, depth)
        board = board[:, :ROWS * COLS, :]  # (batch_size, 64, NUM_MOVES_PER_SQUARE)
        output_policy = self.policy(board)  # (batch_size, 64, NUM_MOVES_PER_SQUARE)
        output_policy = tf.clip_by_value(output_policy, -40, 40)
        output_policy = tf.reshape(
            output_policy, [-1, ROWS, COLS, NUM_MOVES_PER_SQUARE]
        )
        output_q = self.q(q)
        return output_policy, output_q


b = tf.ones([1, 8, 8], dtype=tf.int32)
m = tf.ones([1, 5], dtype=tf.int32)
c = ChessModel(10, 10, 10)
x = c.call(b, m)
