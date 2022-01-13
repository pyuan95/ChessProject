#include "Constants.h"
extern int move2index_initializer(PieceType pt, Move m, Color color) {
	Square f = m.from();
	Square t = m.to();
	int i;
	int diff = color == WHITE ? t - f : f - t;
	switch (pt) {
		int num;
		int dir;
	case KING:
	case QUEEN:
	case ROOK:
	case BISHOP:
		// 0, 1, 2, 3, 4, 5, 6, 7 -> N, NE, E, SE, S, SW, W, NW
		if (diff % NORTH == 0) {
			dir = diff > 0 ? 0 : 4;
			num = diff / NORTH;
		}
		else if (diff % NORTH_EAST == 0) {
			dir = diff > 0 ? 1 : 5;
			num = diff / NORTH_EAST;
		}
		else if (diff % SOUTH_EAST == 0 && rank_of(f) != rank_of(t)) {
			dir = diff > 0 ? 7 : 3;
			num = diff / SOUTH_EAST;
		}
		else if (diff % EAST == 0 && abs(diff) < 8) {
			dir = diff > 0 ? 2 : 6;
			num = diff / EAST;
		}
		else {
			// throw std::runtime_error("Illegal argument provided to move2index: Queen move");
		}
		num = abs(num);
		i = dir * 7 + num - 1;
		break;
	case KNIGHT:
		switch (diff)
		{
		case 10:
			dir = 0;
			break;
		case 17:
			dir = 1;
			break;
		case 15:
			dir = 2;
			break;
		case 6:
			dir = 3;
			break;
		case -10:
			dir = 4;
			break;
		case -17:
			dir = 5;
			break;
		case -15:
			dir = 6;
			break;
		case -6:
			dir = 7;
			break;
		default:
			// throw std::runtime_error("Illegal argument provided to move2index: Knight move");
			break;
		}
		i = dir + 56;
		break;
	case PAWN: {
		bool prom = true;
		switch (m.flags()) {
		case PR_BISHOP:
		case PC_BISHOP:
			i = 64;
			break;
		case PR_KNIGHT:
		case PC_KNIGHT:
			i = 67;
			break;
		case PR_ROOK:
		case PC_ROOK:
			i = 70;
			break;
		default:
			prom = false;
			break;
		}
		switch (diff)
		{
		case 7: // NW
			dir = prom ? 0 : 7;
			break;
		case 8: // N
		case 16: // float push
			dir = prom ? 1 : 0;
			break;
		case 9: // NE
			dir = prom ? 2 : 1;
			break;
		default:
			// throw std::runtime_error("Illegal argument provided to move2index: pawn move");
			break;
		}
		if (!prom) {
			i = dir * 7 + 1 - 1;
			if (diff == 16) // double push
				i += 1;
		}
		else
			i += dir;
		break;
	}
	default:
		// throw std::runtime_error("Invalid PieceType, should never happen!");
		break;
	}
	return i;
}

extern void init_move2index_cache() {
	for (int c = 0; c < NCOLORS; c++) {
		for (int pt = 0; pt < NPIECE_TYPES; pt++) {
			for (unsigned int m = 0; m <= 0xffff; m++) {
				int res;
				Color color = static_cast<Color>(c);
				Move move(m);
				PieceType pieceType = static_cast<PieceType>(pt);

				try {
					res = move2index_initializer(pieceType, move, color);
				}
				catch (const std::exception& error) {

				}

				move2index_cache[c][pt][m] = res;
			}
		}
	}
}

Ndarray<int, 3> move2index_cache = Ndarray<int, 3>(
	new int[NCOLORS * NPIECE_TYPES * (0xffff + 1)](),
	new long[3]{ NCOLORS, NPIECE_TYPES, (0xffff + 1) },
	new long[3]{ NPIECE_TYPES * (0xffff + 1), (0xffff + 1), 1 }
);

Ndarray<float, 3> DUMMY_POLICY = Ndarray<float, 3>(
	new float[ROWS * COLS * MOVES_PER_SQUARE](),
	new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
	new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
);