#include "Position.h"
#include "MCTS.h"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include "PriorityQueue.h"

template<Color color>
static bool writeLegalMoves(Position& p, int moves[ROWS][COLS][MOVES_PER_SQUARE], bool fillzeros) {
	MoveList<color> l(p);
	if (l.size() == 0) // no moves available, this is a terminal position.
		return false;

	for (Move m : l)
	{
		PolicyIndex i = move2index(p, m, color);
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

void testMCTSbitlogic() {
	float prob = 0.32f;
	Move m = Move();
	MCTSNode n = MCTSNode(prob, m, WHITE);
	cout << n.get_prob() << "\n";
	assert(n.get_color() == WHITE);
	assert(!n.is_terminal_position());
	n.mark_terminal_position();
	assert(n.is_terminal_position());

	n = MCTSNode(prob, m, BLACK);
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
	writePosition<WHITE>(p1, board1);
	writePosition<BLACK>(p2, board2);
	for (int i = 0; i < 64; i++)
		assert(board1[i >> 3][i & 8] == board2[i >> 3][i & 8]);
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
			PolicyIndex pi = move2index(pos, m, WHITE);
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
		PolicyIndex pi = move2index(pawn_pos, m, WHITE);
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
		PolicyIndex pi = move2index(pawn_pos, m, WHITE);
		count[pi.i] += 1;
	}
	assert(count[0] == 2);
	assert(count[1] == 1); // float_t push
}

void select_and_update_no_errors() {
	Position p1 = *(new Position());
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
	int x = 0;
	for (int i = 0; i < 1000; i++)
	{
		m->select(10, board);
		m->update(0.0, dummy_policy);
		// std::cout << "completed iter " << i << "\n";
		// std::cout << "root count: " << m->current_sims() << "\n";
		// std::cout << "position: " << m->position() << "\n";
	}
	assert(!m->reached_sim_limit());
	assert(m->move_num() == 1);
	assert(m->current_sims() >= 1000);
	assert(m->position() == p1);

	for (int i = 0; i < 10000; i++) // play 10k more moves. we should be over the max_sim
	{
		m->select(10, board);
		m->update(0.0, dummy_policy);
	}
	assert(m->reached_sim_limit());
	assert(m->move_num() == 1);
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
	while (m->current_sims() < SIMS) {
		m->select(CPUCT, board);
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
	assert(m->get_best_move(0.1).from() == b7); // assert that we found the move that leads to mate in 5.
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

	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < SIMS; j++) {
			m->select(CPUCT, board);
			m->update(0.00, dummy_policy);
		}
		// cout << m->current_sims() << "\n";
	}
	assert(m->move_num() > 995 || m->games().size() > 0);
	cout << "number of games: " << m->games().size() << "\n";
	cout << "move num in current game: " << m->move_num() << "\n";
	m->serialize_games("C://Users//patri//Desktop//Chess Project//Chess Project//Chess Project//SavedGames//games.txt");
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

	while (m->current_sims() < SIMS) {
		m->select(CPUCT, board);
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
	int SIMS = 1000000;
	float CPUCT = 0.01;
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

	while (m->current_sims() < SIMS) {
		m->select(CPUCT, board);
		m->update(0.00, dummy_policy);
		if (m->current_sims() % 100000 == 0)
			std::cout << "Current sims: " << m->current_sims() << "\n";
	}
	std::cout << "Size of MCTSNode: " << sizeof(MCTSNode) << "\n";
	std::cout << "Number of nodes in MCTS Tree: " << m->size() << "\n";
	std::cout << "Number of sims ran: " << m->current_sims() << "\n";
}

void sizeof_node_test() {
	cout << sizeof(Test);
}

void run_all_tests() {
	/*
	print_test(&testMCTSbitlogic, "MCTSNode bit logic test");
	print_test(&test_prio_queue, "Priority Queue Test");
	print_test(&rotation_test, "Rotation Test");
	print_test(&policy_completeness_test, "Policy Completeness Test");
	print_test(&policy_rotation_test, "Policy Rotation Test");
	print_test(&select_and_update_no_errors, "Select and Update no Errors Test");
	print_test(&select_best_move_test, "Select Best Move Test");
	print_test(&autoplay_test, "Autoplay Test");
	print_test(&promotion_test, "Promotion Test");
	print_test(&memory_test, "Memory Test");
	*/
	print_test(&sizeof_node_test, "Size of node test");
}