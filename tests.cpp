#include "tests.h"
#include "MCTS.h"
#include <iostream>
#include <algorithm>
#include "PriorityQueue.h"
#include <unordered_set>
#include "BatchMCTS.h"

template<Color color>
static bool writeLegalMoves(Position& p, int moves[ROWS][COLS][MOVES_PER_SQUARE], bool fillzeros) {
	MoveList<color> l(p);
	if (l.size() == 0) // no moves available, this is a terminal position.
		return false;

	PolicyIndex i;
	for (Move m : l) {
		move2index(p, m, color, i);
		moves[i.r][i.c][i.i] = 1;
	}
	if (fillzeros)
	{
		for (int r = 0; r < ROWS; r++)
		{
			for (int c = 0; c < COLS; c++)
			{
				for (int i = 0; i < MOVES_PER_SQUARE; i++)
					if (moves[r][c][i] != 1)
						moves[r][c][i] = 0;
			}
		}
	}
	return true;
}

void test_move_set() {
	MCTSNode m(WHITE);
	Position p;
	MoveList<WHITE> moves(p);
	unordered_set<uint16_t> stored_moves;
	for (Move m : moves) {
		stored_moves.insert(m.get_representation());
	}
	std::vector<std::pair<Move, float>> leaves(218, pair<Move, float>(0, 0.0f));
	Ndarray<float, 3> policy = Ndarray<float, 3>(
		new float[ROWS * COLS * MOVES_PER_SQUARE](),
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			for (int k = 0; k < MOVES_PER_SQUARE; k++) {
				policy[i][j][k] = 10.0f * std::rand() / RAND_MAX;
			}
		}
	}

	m.expand(p, policy, moves.begin(), moves.size(), leaves);
	assert(m.get_num_children() == moves.size());

	uint8_t prev = 0xff;
	float tot = 0;
	for (int i = 0; i < m.get_num_children(); i++) {
		Move stored(((uint16_t*)(m.begin_children() + i * 3L))[0]);
		uint8_t prob((m.begin_children() + i * 3L)[2]);
		assert(prev >= prob);
		prev = prob;
		// cout << (unsigned int) prev << "\t";
		stored_moves.erase(stored.get_representation());
		tot += (prev + 0.5f) / 256.0f;
	}
	// cout << tot;
	assert(stored_moves.empty());
}

void testMCTSbitlogic() {
	float prob = 0.32f;
	Move m = Move();
	MCTSNode n(WHITE);
	assert(n.get_color() == WHITE);
	assert(!n.is_terminal_position());
	n.mark_terminal_position();
	assert(n.is_terminal_position());

	n = MCTSNode(BLACK);
	assert(n.get_color() == BLACK);
	assert(!n.is_terminal_position());
	n.mark_terminal_position();
	assert(n.is_terminal_position());
}

void test_prio_queue() {
	PriorityQueue<int> q;
	q.push(5, 5.0);
	q.push(6, 6.0);
	assert(q.size() == 2);
	assert(q.peek() == 6);
	q.push(2, 7.0);
	assert(q.peek() == 2);
	q.push(1, 1.0);
	q.push(0, 1.0);
	q.push(2, 7.0);
	q.update_root(-10);
	assert(q.peek() == 2); // we have two 2s
	assert(q.size() == 6);
	q.update_root(-10);
	assert(q.peek() == 6);
	q.update_root(-10);
	assert(q.peek() == 5);
}

void print_test(void (*func)(), string name)
{
	cout << "running test: " << name << "\n";
	func();
	cout << "running test: " << name << " completed\n\n";
}


void rotation_test() {
	Position p1;
	Position p2;
	Position::set("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w - - 0 1", p2);
	// cout << "SIZE: " << sizeof(p1);
	Ndarray<int, 2> board1(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);
	Ndarray<int, 2> board2(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);
	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);
	writePosition<WHITE>(p1, board1, metadata);
	writePosition<BLACK>(p2, board2, metadata);
	for (int i = 0; i < 64; i++)
		assert(board1[i >> 3][i & 8] == board2[i >> 3][i & 8]);
}

void test_metadata() {
	MCTS* m = new MCTS(10000, 0, false);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}
	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	
	assert(metadata[0] + metadata[1] + metadata[2] + metadata[3] == 4);
	assert(metadata[4] == -1);

	Position p;
	p.play<WHITE>(Move("e2e3"));
	p.play<BLACK>(Move("e7e6"));
	p.play<WHITE>(Move("f1e2"));
	p.play<BLACK>(Move("f8e7"));
	p.play<WHITE>(Move("g1f3"));
	p.play<BLACK>(Move("g8f6"));
	p.play<WHITE>(Move("h1g1"));
	m->set_position(p);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	assert(metadata[0] == 1);
	assert(metadata[1] == 1);
	assert(metadata[2] == 0);
	assert(metadata[3] == 1);
	assert(metadata[4] == -1);
	p.play<BLACK>(Move("e8f8"));
	m->set_position(p);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	assert(metadata[0] == 0);
	assert(metadata[1] == 1);
	assert(metadata[2] == 0);
	assert(metadata[3] == 0);
	assert(metadata[4] == -1);
	p.play<WHITE>(Move("h2h4"));
	p.play<BLACK>(Move("b8c6"));
	p.play<WHITE>(Move("h4h5"));
	p.play<BLACK>(Move("d7d5"));
	p.play<WHITE>(Move("b1c3"));
	p.play<BLACK>(Move("d8d6"));
	p.play<WHITE>(Move("d2d4"));
	uint16_t double_push_rep = Move("g7g5").get_representation();
	double_push_rep = double_push_rep | (1 << 12); // move flags: it's a double push!
	Move double_push(double_push_rep);

	p.play<BLACK>(double_push);
	m->set_position(p);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	assert(metadata[0] == 0);
	assert(metadata[1] == 1);
	assert(metadata[2] == 0);
	assert(metadata[3] == 0);
	assert(metadata[4] == 46);
	p.play<WHITE>(Move("d1c3"));
	m->set_position(p);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	assert(metadata[0] == 0);
	assert(metadata[1] == 0);
	assert(metadata[2] == 0);
	assert(metadata[3] == 1);
	assert(metadata[4] == -1);
	p.play<BLACK>(Move("c8d7"));
	p.play<WHITE>(Move("c1d2"));

	m->set_position(p);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	assert(metadata[0] == 0);
	assert(metadata[1] == 0);
	assert(metadata[2] == 0);
	assert(metadata[3] == 1); // p2 (white) has queen side castling
	assert(metadata[4] == -1);

	p.play<BLACK>(Move("d6h2"));
	p.play<WHITE>(Move(e1, a1, MoveFlags(0b0011)));
	m->set_position(p);
	m->select(10, board, metadata);
	m->update(0.0, dummy_policy);
	assert(metadata[0] == 0);
	assert(metadata[1] == 0);
	assert(metadata[2] == 0);
	assert(metadata[3] == 0); // no more castling!
	assert(metadata[4] == -1);
	std::cout << p << "\n";
}

void policy_rotation_test() {
	Position p1;
	Position p2;
	Position::set("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w - - 0 1", p2);
	int policyone[8][8][73] = { {{0}} };
	int policytwo[8][8][73] = {{ {0} }};
	writeLegalMoves<WHITE>(p1, policyone, false);
	writeLegalMoves<BLACK>(p2, policytwo, false);
	for (int r = 0; r < 8; r++)
	{
		for (int c = 0; c < 8; c++)
		{
			for (int i = 0; i < 73; i++) {
				assert(policyone[r][c][i] == policytwo[r][c][i]);
			}
		}
	}
}

void policy_completeness_test() {
	int count[MOVES_PER_SQUARE] = { 0 };
	vector<Position> p;
	for (int i = 0; i < 5; i++)
		p.push_back(Position());
	Position::set("8/8/8/8/8/8/8/Q7 w - - 0 1", p[0]);
	Position::set("7Q/8/8/8/8/8/8/8 w - - 0 1", p[1]);
	Position::set("B7/8/8/8/8/8/8/8 w - - 0 1", p[2]);
	Position::set("8/8/8/8/8/8/8/7B w - - 0 1", p[3]);
	Position::set("8/8/8/3N4/8/8/8/8 w - - 0 1", p[4]);
	for (Position pos : p)
	{
		// cout << pos << "\n";
		for (Move m : MoveList<WHITE>(pos))
		{
			// cout << m << "\n";
			PolicyIndex pi;
			move2index(pos, m, WHITE, pi);
			count[pi.i] += 1;
		}
	}
	for (int i = 0; i < 64; i++) {
		// cout << i << "\n";
		// cout << count[i] << "\n\n";
		assert(count[i] == 1);
	}
	for (int i = 64; i < 73; i++)
		assert(count[i] == 0);

	Position pawn_pos;
	Position::set("2n1n3/3P4/8/8/8/8/8/8 w - - 0 1", pawn_pos);
	for (int i = 0; i < 73; i++)
		count[i] = 0;
	for (Move m : MoveList<WHITE>(pawn_pos))
	{
		// cout << m << "\n";
		PolicyIndex pi;
		move2index(pawn_pos, m, WHITE, pi);
		count[pi.i] += 1;
	}

	for (int i = 64; i < 73; i++)
	{
		// cout << i << "\n";
		// cout << count[i] << "\n\n";
		assert(count[i] == 1);
	}
	for (int i : {0, 7, 49})
	{
		assert(count[i] == 1);
	}
	int tot = 0;
	for (int i = 0; i < 73; i++)
		tot += count[i];
	assert(tot == 12);

	// finally test the float_t pawn push
	
	Position::set("8/8/8/8/8/8/3P4/8 w - - 0 1", pawn_pos);
	for (Move m : MoveList<WHITE>(pawn_pos))
	{
		PolicyIndex pi;
		move2index(pawn_pos, m, WHITE, pi);
		count[pi.i] += 1;
	}
	assert(count[0] == 2);
	assert(count[1] == 1); // float_t push
}

void select_and_update_no_errors() {
	Position p1;
	MCTS* m = new MCTS(10000, 0, false);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}
	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);
	int x = 0;
	for (int i = 0; i < 1000; i++)
	{
		m->select(10, board, metadata);
		m->update(0.0, dummy_policy);
		// std::cout << "completed iter " << i << "\n";
		// std::cout << "root count: " << m->current_sims() << "\n";
		// std::cout << "position: " << m->position() << "\n";
	}
	assert(!m->reached_sim_limit());
	assert(m->move_number() == 1);
	assert(m->current_sims() >= 1000);
	assert(m->position() == p1);

	for (int i = 0; i < 10000; i++) // play 10k more moves. we should be over the max_sim
	{
		m->select(10, board, metadata);
		m->update(0.0, dummy_policy);
	}
	assert(m->reached_sim_limit());
	assert(m->move_number() == 1);
	assert(m->current_sims() == 10000);
	assert(m->position() == p1);
	delete m;
}

bool comp(pair<Move, float_t> x, pair<Move, float_t> y) { return x.second > y.second; } // we want to sort in reverse order.

void select_best_move_test() {
	const int SIMS = 100000;
	const float_t CPUCT = 0.5; // pretty wide search!

	MCTS* m = new MCTS(SIMS, 1.0, false);
	Position p;
	Position::set("2r3k1/1b3pp1/p3p2p/2b1P2P/5PPK/1NPr4/PP1p4/3R1R2 b - - 0 1", p);
	m->set_position(p);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}

	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);
	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);
	while (m->current_sims() < SIMS) {
		m->select(CPUCT, board, metadata);
		m->update(0.00, dummy_policy);
		if (m->current_sims() % 100000 == 0)
			std::cout << "Current sims: " << m->current_sims() << "\n";
	}
	std::cout << "Size of MCTSNode: " << sizeof(MCTSNode) << "\n";
	std::cout << "Number of nodes in MCTS Tree: " << m->size() << "\n";
	std::cout << "Number of sims ran: " << m->current_sims() << "\n";
	vector<pair<Move, float_t>> pol = m->policy(1);
	std::sort(pol.begin(), pol.end(), comp);

	for (auto k : pol)
	{
		std::cout << k.first << "\t" << k.second << "\n";
	}
	// std::cout << (m->turn() ? "BLACK" : "WHITE") << "'s Q evaluation: " << m->evaluation() << "\n";
	// std::cout << (m->turn() ? "BLACK" : "WHITE") << "'s minimax evaluation: " << m->minimax_evaluation() << "\n";
	assert(m->get_best_move(0.1f).from() == b7); // assert that we found the move that leads to mate in 5.
	delete m;
}

void autoplay_test() {
	const int SIMS = 250;
	const float_t CPUCT = 0.5; // pretty wide search!

	MCTS* m = new MCTS(SIMS, 1.0, true);
	Position p;
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);



	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}

	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);

	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < SIMS; j++) {
			m->select(CPUCT, board, metadata);
			m->update(0.00, dummy_policy);
		}
		// cout << m->current_sims() << "\n";
	}
	assert(m->move_number() > 995 || m->game_number() > 1);
	cout << "number of games: " << m->game_number() << "\n";
	cout << "move num in current game: " << m->move_number() << "\n";
	delete m;
}

void promotion_test() {
	const int SIMS = 10000;
	const float_t CPUCT = 0.5; // pretty wide search!

	MCTS* m = new MCTS(SIMS, 1.0, false);
	Position p;
	Position::set("6k1/8/8/8/8/8/8/R3K2R w KQ - 0 1", p);
	std::cout << p << "\n";
	m->set_position(p);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}

	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);

	while (m->current_sims() < SIMS) {
		m->select(CPUCT, board, metadata);
		m->update(0.00, dummy_policy);
		if (m->current_sims() % 100000 == 0)
			std::cout << "Current sims: " << m->current_sims() << "\n";
	}
	// std::cout << "Size of MCTSNode: " << sizeof(MCTSNode) << "\n";
	// std::cout << "Number of nodes in MCTS Tree: " << m->size() << "\n";
	// std::cout << "Number of sims ran: " << m->current_sims() << "\n";
	vector<pair<Move, float_t>> pol = m->policy(1);
	std::sort(pol.begin(), pol.end(), comp);

	for (auto k : pol)
	{
		std::cout << k.first << "\t" << k.second << "\n";
	}
	// std::cout << (m->turn() ? "BLACK" : "WHITE") << "'s Q evaluation: " << m->evaluation() << "\n";
	// std::cout << (m->turn() ? "BLACK" : "WHITE") << "'s minimax evaluation: " << m->minimax_evaluation() << "\n";
	// assert(m->get_best_move(0.1).from() == b7); // assert that we found the move that leads to mate in 5.
	delete m;
}

void memory_test() {
	for (int abc = 0; abc < 3; abc++) {
		int SIMS = 1000000;
		float CPUCT = 0.01f;
		MCTS* m = new MCTS(1000000, 1.0, false);
		Ndarray<float, 3> dummy_policy(
			new float[ROWS * COLS * MOVES_PER_SQUARE],
			new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
			new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
		);
		for (int r = 0; r < ROWS; r++)
		{
			for (int c = 0; c < COLS; c++)
			{
				for (int i = 0; i < MOVES_PER_SQUARE; i++)
					dummy_policy[r][c][i] = 0.10f;
			}
		}

		Ndarray<int, 2> board(
			new int[ROWS * COLS],
			new long[2]{ ROWS, COLS },
			new long[2]{ COLS, 1 }
		);

		Ndarray<int, 1> metadata(
			new int[METADATA_LENGTH],
			new long[1]{ METADATA_LENGTH },
			new long[1]{ 1 }
		);

		while (m->current_sims() < SIMS) {
			m->select(CPUCT, board, metadata);
			m->update(0.00, dummy_policy);
			if (m->current_sims() % 100000 == 0)
				std::cout << "Current sims: " << m->current_sims() << "\n";
		}
		std::cout << "Size of MCTSNode: " << sizeof(MCTSNode) << "\n";
		std::cout << "Number of nodes in MCTS Tree: " << m->size() << "\n";
		std::cout << "Number of sims ran: " << m->current_sims() << "\n";

		delete m;
	}
}

void test_next_move_randomness() {
	int SIMS = 100000;
	float CPUCT = 0.01f;
	MCTS* m = new MCTS(1000000, 1.0, false);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}

	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);

	while (m->current_sims() < SIMS) {
		m->select(CPUCT, board, metadata);
		m->update(0.00f, dummy_policy);
	}

	unordered_set<uint16_t> captured_moves;
	for (unsigned int i = 0; i < 1000000; i++) {
		captured_moves.insert(m->get_best_move(1.0).get_representation());
	}
	Position pos;
	assert(captured_moves.size() == MoveList<WHITE>(pos).size());
	delete m;
}

void test_updated_q() {
	int SIMS = 1000000;
	float CPUCT = 0.01f;
	MCTS* m = new MCTS(1000000, 1.0, false);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	Ndarray<int, 2> board(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);

	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			for (int i = 0; i < MOVES_PER_SQUARE; i++)
				dummy_policy[r][c][i] = 0.10f;
		}
	}

	m->select(1, board, metadata);
	m->update(150, dummy_policy);
	m->select(0, board, metadata);
	m->update(-2.5, dummy_policy);
	assert(std::abs(m->evaluation() - ((150.0f + 2.5f) / 2.0f)) < 0.001);
	assert(std::abs(m->minimax_evaluation() - 2.5f) < 0.001);
	delete m;
}

void test_select_best_move_correctly() {
	MCTSNode m(WHITE);
	Position p;
	MoveList<WHITE> moves(p);

	std::vector<std::pair<Move, float>> leaves(218, pair<Move, float>(0, 0.0f));
	Ndarray<float, 3> policy = Ndarray<float, 3>(
		new float[ROWS * COLS * MOVES_PER_SQUARE](),
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			for (int k = 0; k < MOVES_PER_SQUARE; k++) {
				policy[i][j][k] = 10.0f * std::rand() / RAND_MAX;
			}
		}
	}

	m.expand(p, policy, moves.begin(), moves.size(), leaves);
	std::pair<MCTSNode*, Move> child(0, 0);
	m.select_best_child(0.01f, child);
	child.first->backup(-100.0f); // really good for us now
	MCTSNode* prev_best = child.first;
	m.select_best_child(0.01f, child);
	assert(child.first == prev_best);

	child.first->backup(1000); // now really bad for us; next node should be a new one;
	prev_best = child.first;
	m.select_best_child(0.01f, child);
	assert(child.first == ((MCTSNode*) m.begin_nodes()) + 1);

	child.first->backup(1000); // now really bad for us; next node should be a new one;
	prev_best = child.first;
	m.select_best_child(0.01f, child);
	assert(child.first == ((MCTSNode*)m.begin_nodes()) + 2);

	child.first->backup(-1000); // now really good for us; next node should be the same one;
	prev_best = child.first;
	m.select_best_child(0.01f, child);
	m.select_best_child(0.01f, child);
	m.select_best_child(0.01f, child);
	assert(child.first == prev_best);
}

void batch_mcts_test() {
	int iterations = 5000;
	int num_sims_per_move = 1600;
	float temperature = 1.0;
	bool autoplay = true;
	string output = "./output/output";
	output = "";

	int num_threads = 8;
	int batch_size = 1000;
	int num_sectors = 2;
	float cpuct = 1.0;

	Ndarray<int, 3> boards(
		new int[batch_size * num_sectors * ROWS * COLS],
		new long[3]{ batch_size * num_sectors, ROWS, COLS },
		new long[3]{ ROWS * COLS, COLS, 1 }
	);

	Ndarray<float, 4> policy(
		new float[batch_size * ROWS * COLS * MOVES_PER_SQUARE](),
		new long[4]{ batch_size, ROWS, COLS, MOVES_PER_SQUARE },
		new long[4]{ ROWS * COLS * MOVES_PER_SQUARE, COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);

	Ndarray<float, 1> q(
		new float[batch_size],
		new long[1]{ batch_size },
		new long[1]{ 1 }
	);

	Ndarray<int, 2> metadata(
		new int[batch_size * num_sectors * METADATA_LENGTH],
		new long[2]{ batch_size * num_sectors, METADATA_LENGTH },
		new long[2]{ METADATA_LENGTH, 1 }
	);

	for (int a = 0; a < batch_size; a++) {
		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLS; j++) {
				for (int k = 0; k < MOVES_PER_SQUARE; k++) {
					policy[a][i][j][k] = 10.0f * std::rand() / RAND_MAX;
				}
			}
		}
	}

	BatchMCTS m(
		num_sims_per_move, temperature, autoplay, output, num_threads, batch_size, num_sectors, cpuct, boards, metadata
	);

	using namespace std;
	using namespace std::chrono;
	auto start = high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {
		// std::cout << i << "\n";
		m.select();
		m.update(q, policy);
		if (i % 100 == 0)
			std::cout << i << "\n";
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "speed: " << iterations * batch_size * 1000.0 / duration.count() << " sims per second\n";
	cout << "speed: " << iterations * batch_size * 1000.0 / duration.count() / num_threads << " sims per second per thread\n";

	int SIMS = 1000000;
	float CPUCT = 0.01f;
	MCTS* mcts = new MCTS(1000000, 1.0, false);
	Ndarray<float, 3> dummy_policy(
		new float[ROWS * COLS * MOVES_PER_SQUARE],
		new long[3]{ ROWS, COLS, MOVES_PER_SQUARE },
		new long[3]{ COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);
	Ndarray<int, 2> board_baseline(
		new int[ROWS * COLS],
		new long[2]{ ROWS, COLS },
		new long[2]{ COLS, 1 }
	);

	Ndarray<int, 1> metadata_baseline(
		new int[METADATA_LENGTH],
		new long[1]{ METADATA_LENGTH },
		new long[1]{ 1 }
	);

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			for (int k = 0; k < MOVES_PER_SQUARE; k++) {
				dummy_policy[i][j][k] = 10.0f * std::rand() / RAND_MAX;
			}
		}
	}

	start = high_resolution_clock::now();
	for (int i = 0; i < 100000; i++) {
		mcts->select(CPUCT, board_baseline, metadata_baseline);
		mcts->update(0.0f, dummy_policy);
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	cout << "baseline speed: " << 1000.0f * 100000 / duration.count() << " sims per second\n";
}

void test_ndarray_copy() {
	int batch_size = 100;

	Ndarray<float, 4> policy(
		new float[batch_size * ROWS * COLS * MOVES_PER_SQUARE](),
		new long[4]{ batch_size, ROWS, COLS, MOVES_PER_SQUARE },
		new long[4]{ ROWS * COLS * MOVES_PER_SQUARE, COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);

	Ndarray<float, 4> policy2(
		new float[batch_size * ROWS * COLS * MOVES_PER_SQUARE](),
		new long[4]{ batch_size, ROWS, COLS, MOVES_PER_SQUARE },
		new long[4]{ ROWS * COLS * MOVES_PER_SQUARE, COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
	);

	for (int a = 0; a < batch_size; a++) {
		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLS; j++) {
				for (int k = 0; k < MOVES_PER_SQUARE; k++) {
					policy[a][i][j][k] = 10.0f * std::rand() / RAND_MAX;
				}
			}
		}
	}

	policy2.copy(policy);

	for (int a = 0; a < batch_size; a++) {
		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLS; j++) {
				for (int k = 0; k < MOVES_PER_SQUARE; k++) {
					assert(policy[a][i][j][k] == policy2[a][i][j][k]);
				}
			}
		}
	}
}

void run_all_tests() {
	print_test(&test_metadata, "metadata test");
	print_test(&batch_mcts_test, "batch mcts");
	if (false) {
		// print_test(&batch_mcts_test, "batch mcts");
		print_test(&promotion_test, "Promotion Test");
		print_test(&test_ndarray_copy, "test ndarray copy");
		print_test(&autoplay_test, "Autoplay Test");
		print_test(&test_select_best_move_correctly, "MCTSNode select best child test");
		print_test(&test_next_move_randomness, "text next move randomness test");
		print_test(&test_updated_q, "updated q test");
		print_test(&select_best_move_test, "Select Best Move Test");
		print_test(&test_move_set, "test MCTSNode setting moves");
		print_test(&testMCTSbitlogic, "MCTSNode bit logic test");
		print_test(&test_prio_queue, "Priority Queue Test");
		print_test(&rotation_test, "Rotation Test");
		print_test(&policy_completeness_test, "Policy Completeness Test");
		print_test(&policy_rotation_test, "Policy Rotation Test");
		print_test(&select_and_update_no_errors, "Select and Update no Errors Test");
		print_test(&memory_test, "Memory Test");
	}
}