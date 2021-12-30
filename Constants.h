#pragma once
#include <math.h>
#include "ndarray.h"

const int MOVES_PER_SQUARE = 73;
const float EPSILON = pow(10.0f, -20.0f);
const int ROWS = 8;
const int COLS = 8;
const int MOVE_SIZE = ROWS * COLS * MOVES_PER_SQUARE;
const bool FILL_ZEROS = false; // whether to fill in zeros in the legal moves. may have to change later based on implementation.
const Ndarray<float, 3> DUMMY_POLICY = Ndarray<float, 3>(
	new float[ROWS * COLS * MOVES_PER_SQUARE](),
	new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
	new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
);
const float_t DUMMY_Q = 0.0;