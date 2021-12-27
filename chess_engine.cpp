#include <iostream>
#include <chrono>
#include "tables.h"
#include "position.h"
#include "types.h"
#include <iostream>


//Computes the perft of the position for a given depth, using bulk-counting
//According to the https://www.chessprogramming.org/Perft site:
//Perft is a debugging function to walk the move generation tree of strictly legal moves to count 
//all the leaf nodes of a certain depth, which can be compared to predetermined values and used to isolate bugs
template<Color Us>
unsigned long long perft(Position& p, unsigned int depth) {
	unsigned long long nodes = 0;

	MoveList<Us> list(p);

	if (depth == 1) return (unsigned long long) list.size();

	for (Move move : list) {
		p.play<Us>(move);
		nodes += perft<~Us>(p, depth - 1);
		p.undo<Us>(move);
	}

	return nodes;
}

//A variant of perft, listing all moves and for each move, the perft of the decremented depth
//It is used solely for debugging
template<Color Us>
void perftdiv(Position& p, unsigned int depth) {
	unsigned long long nodes = 0, pf;

	MoveList<Us> list(p);

	for (Move move : list) {
		std::cout << move;

		p.play<Us>(move);
		pf = perft<~Us>(p, depth - 1);
		std::cout << ": " << pf << " moves\n";
		nodes += pf;
		p.undo<Us>(move);
	}

	std::cout << "\nTotal: " << nodes << " moves\n";
}

void test_perft() {
	Position p;
	Position::set("rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq -", p);
	std::cout << p;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	auto n = perft<WHITE>(p, 6);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	auto diff = end - begin;

	std::cout << "Nodes: " << n << "\n";
	std::cout << "NPS: "
		<< int(n * 1000000.0 / std::chrono::duration_cast<std::chrono::microseconds>(diff).count())
		<< "\n";
	std::cout << "Time difference = "
		<< std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << " [microseconds]\n";
}

void draw_test() {
	Position p;
	std::cout << p;
	p.play<WHITE>(Move("e2e4"));
	p.play<BLACK>(Move("e7e5"));
	for (int i = 0; i < 2; i++) {
		p.play<WHITE>(Move("e1e2"));
		p.play<BLACK>(Move("e8e7"));
		p.play<WHITE>(Move("e2e1"));
		p.play<BLACK>(Move("e7e8"));
	}
	std::cout << p;
	MoveList<WHITE> list(p);
	list.size();                         
}

int demo() {
	initialise_all_databases();
	zobrist::initialise_zobrist_keys();

	Position p;
	std::cout << p;
	p.play<WHITE>(Move("e2e4"));
	p.play<BLACK>(Move("e7e5"));
	std::cout << "\n\n\n" << p.num_ply_no_capture_or_pawn_move() << "\n";
	for (int i = 0; i < 2; i++) {
		p.play<WHITE>(Move("e1e2"));
		p.play<BLACK>(Move("e8e7"));
		p.play<WHITE>(Move("e2e1"));
		p.play<BLACK>(Move("e7e8"));
	}

	std::cout << p;
	MoveList<WHITE> list(p);
	for (Move m : list) {
		std::cout << m << "\n";
	}

	std::cout << "after undoing move...\n\n";
	p.undo<BLACK>(Move("e7e8"));

	std::cout << p;
	MoveList<BLACK> listb(p);
	for (Move m : listb) {
		std::cout << m << "\n";
	}

	std::cout << "doing random move\n\n";
	p.play<BLACK>(Move("a7a6"));
	std::cout << p;
	list = MoveList<WHITE>(p);
	for (Move m : list) {
		std::cout << m << "\n";
	}

	std::cout << "should be draw...\n";
	p.undo<BLACK>(Move("a7a6"));
	p.play<BLACK>(Move("e7e8"));
	std::cout << p;
	list = MoveList<WHITE>(p);
	for (Move m : list) {
		std::cout << m << "\n";
	}

	std::cout << "playing 50 more non pawn move or captures, counter should be incremented to 108\n";

	for (int i = 0; i < 25; i++) {
		p.play<WHITE>(Move("e1e2"));
		p.play<BLACK>(Move("e8e7"));
		p.play<WHITE>(Move("e2e1"));
		p.play<BLACK>(Move("e7e8"));
	}

	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << "undoing one move, counter should be at 107 now.\n";
	p.undo<BLACK>(Move("e7e8"));
	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << "playing a knight move, counter should be 108.\n";
	p.play<BLACK>(Move("b8c6"));
	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << "playing a pawn move, counter should be 0.\n";
	p.play<WHITE>(Move("a2a3"));
	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << p << "\n";

	p.play<BLACK>(Move("c8b6"));
	p.play<WHITE>(Move("e1e2"));
	p.play<BLACK>(Move("e7e8"));
	p.play<WHITE>(Move("e2e3"));
	p.play<BLACK>(Move(Square(0), Square(0), OO));
	std::cout << "counter should be 5.\n";
	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << p << "\n";

	std::cout << "capturing a piece. counter should be 0\n";
	p.play<WHITE>(Move(e4, e5, CAPTURE));
	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << p << "\n";

	std::cout << "undoing the capture. counter should be 5\n";
	p.undo<BLACK>(Move(Square(0), Square(0), OO));
	std::cout << p.num_ply_no_capture_or_pawn_move() << "\n";
	std::cout << p << "\n";

	return 0;
}
