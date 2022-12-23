#pragma once
#include <iostream>
#include <math.h>
#include "ndarray.h"
#include "types.h"

extern int move2index_initializer(PieceType pt, Move m, Color color);
const int METADATA_LENGTH = 5;
const int DRAWING_MOVE_COUNT = 50;
const int MAX_MOVES = 218;
const int MOVES_PER_SQUARE = 73;
const float EPSILON = pow(10.0f, -20.0f);
const int ROWS = 8;
const int COLS = 8;
const int MOVE_SIZE = ROWS * COLS * MOVES_PER_SQUARE;
const bool FILL_ZEROS = false; // whether to fill in zeros in the legal moves. may have to change later based on implementation.
extern Ndarray<float, 3> DUMMY_POLICY;
const float_t DUMMY_Q = 0.0;
extern Ndarray<int, 3> move2index_cache;

extern void init_move2index_cache();
