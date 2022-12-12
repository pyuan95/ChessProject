#include <iostream>
#include "tests.h"
#include "MCTS.h"
#include <vector>
using namespace std;

int main() {
    init_rand();
    initialise_all_databases();
	zobrist::initialise_zobrist_keys();
	init_move2index_cache();
	run_all_tests();
	return 0;
}