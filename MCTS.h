#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <fstream>
#include <unordered_set>
#include "Constants.h"
#include "position.h"
#include "tables.h"
#include "float.h"
using namespace std;

// stores the indices in the policy array
struct PolicyIndex {
	int r;
	int c;
	int i;

	PolicyIndex(const int& r, const int& c, const int& i) : r(r), c(c), i(i) {}
	PolicyIndex() {}
};

struct LegalMoves {
	int l[ROWS][COLS][MOVES_PER_SQUARE] = { 0 };
};

// wrapper class for the policy array; we can't directly store arrays in a vector.
struct Policy {
	float p[ROWS][COLS][MOVES_PER_SQUARE] = { 0.0f };
};

// wrapper class for the board state; we can't directly store arrays in a vector.
struct BoardState {
	int b[ROWS][COLS];
	int m[METADATA_LENGTH];
};

extern ostream& operator<<(ostream& os, const Policy& p);

extern ostream& operator<<(ostream& os, const BoardState& b);

extern ostream& operator<<(ostream& os, const LegalMoves& l);

// each MCTS object will contain a member of vector<Game>, which it will use to store completed games.
struct Game {
	vector<Policy> policy_history;  // policy[i] is the policy for board_history[i]
	vector<BoardState> board_history; // stores the board_history. boards are rotated and pieces are inverted if necessary.
	vector<Move> move_history; // move_history[i] is the move from board_history[i] to board_history[i + 1]
	vector<Color> color_history; // color_history[i] is the color of the turn to move for board_history[i].
	float winner = 2; // -1 for black, 0 for draw, 1 for white, 2 for in progress.
};

inline int invert(int piece) {
	if (piece <= 5)
		return piece + 8;
	else if (piece <= 13)
		return piece - 8;
	return 14;
}

// writes a position to the given board.
// 0...5 = our side (pawn, knight, bishop, rook, queen, king)
// 8...13 = their side (pawn, knight, bishop, rook, queen, king)
// 14 = empty
template<Color color>
inline void writePosition(const Position& p, Ndarray<int, 2>& board, Ndarray<int, 1>& metadata) {
	int piece;
	for (int r = 0; r < ROWS; r++) {
		for (int c = 0; c < COLS; c++) {
			piece = p.at(create_square(File(c), Rank(r)));
			if (color == BLACK) {
				// invert and rotate
				piece = invert(piece);
				board[ROWS - r - 1][COLS - c - 1] = piece;
			}
			else {
				board[r][c] = piece;
			}
		}
	}

	metadata[0] = (p.history[p.ply()].entry & WHITE_OO_MASK) == 0;
	metadata[1] = (p.history[p.ply()].entry & WHITE_OOO_MASK) == 0;
	metadata[2] = (p.history[p.ply()].entry & BLACK_OO_MASK) == 0;
	metadata[3] = (p.history[p.ply()].entry & BLACK_OOO_MASK) == 0;
	metadata[4] = static_cast<int>(p.history[p.ply()].epsq);

	if (color == BLACK) {
		std::swap(metadata[0], metadata[2]);
		std::swap(metadata[1], metadata[3]);
		metadata[4] = 63 - metadata[4];
	}
	if (metadata[4] == 64) // no epsq
		metadata[4] = -1;
}

template<Color color>
inline void writePosition(const Position& p, int board[ROWS][COLS], int metadata[METADATA_LENGTH]) {
	int piece;
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			piece = p.at(create_square(File(c), Rank(r)));
			if (color == BLACK) {
				// invert and rotate
				piece = invert(piece);
				board[ROWS - r - 1][COLS - c - 1] = piece;
			}
			else {
				board[r][c] = piece;
			}
		}
	}

	metadata[0] = (p.history[p.ply()].entry & WHITE_OO_MASK) == 0;
	metadata[1] = (p.history[p.ply()].entry & WHITE_OOO_MASK) == 0;
	metadata[2] = (p.history[p.ply()].entry & BLACK_OO_MASK) == 0;
	metadata[3] = (p.history[p.ply()].entry & BLACK_OOO_MASK) == 0;
	metadata[4] = static_cast<int>(p.history[p.ply()].epsq);

	if (color == BLACK) {
		std::swap(metadata[0], metadata[2]);
		std::swap(metadata[1], metadata[3]);
		metadata[4] = 63 - metadata[4];
	}
	if (metadata[4] == 64) // no epsq
		metadata[4] = -1;
}

void move2index(const Position& p, Move m, Color color, PolicyIndex& policyIndex);

// initializes the random seed with the current time.
void init_rand();

/*
The class that represents a node in the MCTS Tree
*/
class MCTSNode {
private:
	// two bits store itp and color.
	uint8_t color_itp;
	uint8_t num_children;
	uint8_t num_expanded;
	float q;
	uint32_t num_times_selected;
	uint8_t* children;

	inline void set_terminal_position(bool itp) { color_itp = color_itp | ((int)itp); }

	inline void set_color(Color c) { color_itp = color_itp | (c << 1u); }

	inline float convert_prob(uint8_t p) { return (p + 0.5f) / 256.0f; }

	inline void set_prob_at(long long node_num, float prob) { 
		if (!std::isfinite(prob)) {
			cout << "prob is not finite! aborting...";
			throw runtime_error("prob is not finite! aborting...");
		}
		(children + node_num * 3L)[2] = (uint8_t) std::min(prob * 256.0f, 255.0f); 
	}

	inline float get_prob_at(long long node_num) { return convert_prob((children + node_num * 3L)[2]); }

	inline void set_move_at(long long node_num, Move m) { ((uint16_t*)(children + node_num * 3L))[0] = m.get_representation(); }

	inline Move get_move_at(long long node_num) { return ((uint16_t*) (children + node_num * 3L))[0]; }

	inline MCTSNode& get_node_at(int node_num) { return begin_nodes()[node_num]; }

	inline void reallocate_memory() {
		int s = sizeof(uint8_t) * 3 * ((int) num_children) + sizeof(MCTSNode) * std::min(num_expanded + 1, (int)num_children);
		void* new_children = realloc(children, s);
		if (new_children == nullptr) {
			std::cout << "error reallocing children. exiting...";
			free(children);
			exit(1);
		}
		children = (uint8_t*) new_children;
	}

	inline void init_memory() {
		int s = sizeof(uint8_t) * 3 * ((int) num_children);
		children = (uint8_t*) malloc(s);
		if (children == nullptr) {
			std::cout << "error allocating children. exiting...";
			exit(1);
		}
	}

	// adds a leaf to the nodes
	MCTSNode* add_leaf();

public:

	inline size_t get_num_expanded() { return num_expanded; }

	inline size_t get_num_children() { return num_children; }

	// returns true if this node is a leaf. node may or may not be terminal as well.
	inline bool is_leaf() { return num_children == 0; }

	// returns true if this node is a terminal position ( no moves available, game has ended. is not a leaf. )
	inline bool is_terminal_position() { return color_itp & 1u; }

	// returns the color of the side whose turn it is to play
	inline Color get_color() { return (color_itp >> 1) & 1u ? BLACK : WHITE; }

	// marks the current node as terminal position.
	inline void mark_terminal_position() { set_terminal_position(true); }

	// returns the number of times this node was selected.
	inline uint32_t get_num_times_selected() { return num_times_selected; }

	// returns the q value for this node. value is relative to this node's color (1 is good for current color, -1 is bad)
	inline float get_mean_q() { return num_times_selected > 0 ? q / num_times_selected : 0.0f; }

	inline MCTSNode* begin_nodes() { return (MCTSNode*) (children + ((long long) (num_children * 3))); }

	inline MCTSNode* end_nodes() { return begin_nodes() + num_expanded; }

	inline uint8_t* begin_children() { return children; }

	inline uint8_t* begin_leaves() { return children + ((long long)(num_expanded * 3)); }

	inline uint8_t* end_leaves() { return children + ((long long)(num_children * 3)); }

	// the q value must be calculated given the color of the node.
	void backup(float q);

	// expands the node given the policy. the q value must be updated using the update method.
	// requires: the current node is a leaf. If it is terminal, no changes are made.
	// policy is appropriately rotated if the player is black.
	void expand(
		Position& p,
		Ndarray<float, 3>& policy,
		const Move* moves,
		size_t size,
		vector<pair<Move, float>>& leaves
	);

	// returns the number of nodes currently under this tree.
	size_t size();

	// returns the minimax evaluation of the tree rooted at this node.
	float minimax_evaluation();

	// calculates the policy and returns it as a cumulative sum array. res[i] == total sum of children 0....i inclusive.
	// does not include leaves, as leaves have 0 count
	// note that this method does not return a probability distribution; cumsum was not divided by the max.
	// remember to delete it after!
	// no children in leaves are included.
	float* calculate_policy_cumsum(float temperature);

	// calculates the policy and returns it as an array. guaranteed to be a probability distribution.
	// size == children.size()
	// remember to delete it after!
	// no children in leaves are included.
	float* calculate_policy(float temperature);

	// calculates the policy and returns a vector, each element is a pair of Move, Probability
	// all moves are included (including those with 0 probability!)
	vector<pair<Move, float>> policy(float temperature);

	// returns the child with the highest upper bound according to the PUCT algorithm.
	// if it has no children then null is returned.
	void select_best_child(const float cpuct, std::pair<MCTSNode*, Move>& child);

	// returns the child to play based on visit count with the given temperature parameter.
	// if there are no children that are not leaves, nullptr is returned.
	// this can happen when the current node is a leaf (could be terminal position), or if
	// this node was never selected for expansion.
	std::pair<MCTSNode*, Move> select_best_child_by_count(float temperature = 1.0);

	MCTSNode(Color c) :
		color_itp(0),
		num_children(0),
		num_expanded(0),
		q(0.0),
		num_times_selected(0),
		children(0) {
		set_color(c);
	}

	MCTSNode() {} // only used for making the array
};

void recursive_delete(MCTSNode& n, MCTSNode* ignore);

/*
Represents an MCTS Tree
CLass invariant: root -> color == p.turn().
*/
class MCTS {
private:
	MCTSNode* root;
	MCTSNode* best_leaf;
	Move* moves;
	uint64_t nmoves;
	vector<std::pair<MCTSNode*, Move>> best_leaf_path;
	vector<pair<Move, float>> leaves;
	Position p;
	bool auto_play;
	uint32_t sims;
	const float default_temp;
	string output_path_base;
	ofstream output;
	int move_num;
	int game_num;

	// adds a move to the current game
	// we want to pass by value bc board_state, p, m, c, are made on stack.
	void add_move(BoardState& board_state, Policy& p, LegalMoves& legal_moves, Move& m, Color& c);

	// declares a winner for the current game
	void declare_winner(float c);

	// adds a new game and resets all PIVs.
	void new_game();

	// the select helper. returns true if select-helper led to selecting on a terminal position and nothing was written.
	// false otherwise (even if autoplay off and max sims reached)
	// white_moves and black_moves will be set to nullptr if this function returns false.
	// requires: max sims not reached.
	bool select_helper(const float cpuct, Ndarray<int, 2>& board, Ndarray<int, 1>& metadata);

	inline void update_output() {
		if (this->output.is_open())
			this->output.close();
		if (!output_path_base.empty()) {
			this->output.open(output_path_base + "_" + to_string(game_num));
		}
	}

public:

	// the temperature to use for calculating the policy and selecting the best child by count.
	float temperature;

	inline Position position() { return p; } // for debugging only

	inline void set_position(const Position& pos) {
		this->p = pos;
		if (root != nullptr)
			recursive_delete(*root, nullptr);
		if (pos.turn() == WHITE)
			root = new MCTSNode(WHITE);
		else
			root = new MCTSNode(BLACK);
		best_leaf = nullptr;
		best_leaf_path.clear();
		temperature = default_temp;
		nmoves = 0;
		move_num = 1;
		game_num = 1;
	}

	// returns whether we have reached the sim limit. relevent only if auto_play is false.
	inline bool reached_sim_limit() { return root->get_num_times_selected() >= sims; }

	// returns the current number of sims in the root.
	inline int current_sims() { return root->get_num_times_selected(); }

	inline void set_sims(unsigned int count) { sims = count; }

	// returns whose turn it is to play. 0 for white, 1 for black.
	inline int turn() { return p.turn(); }

	// returns the current policy.
	inline vector<pair<Move, float>> policy(float temperature) { return root->policy(temperature); }

	// returns the current player's Q evaluation of the game state (mean q). 1 means we are winning, 0 means
	// game is even, -1 means we are losing.
	inline float evaluation() { return root->get_mean_q(); }

	// returns the current player's minimax evaluation of the game state.
	// 1 means we are winning, 0 means game is even, -1 means we are losing.
	inline float minimax_evaluation() { return root->minimax_evaluation(); }

	// selects the best move by count with the given temperature
	inline Move get_best_move(float temperature) { return root->select_best_child_by_count(temperature).second; }

	// returns the number of nodes in this tree.
	inline size_t size() { return root->size(); }

	// the move number that the current game is on. starts at 1.
	int move_number();

	// the game number we are on. starts at 1.
	int game_number();

	// selects the best leaf thru MCTS and writes the position and the legal moves. Not threadsafe.
	// Additionally, sets the best_leaf* to point to the selected node.
	// if max sims is reached aNdarray<float, 3> DUMMY_POLICY;nd auto play is off, this method does nothing (does not assign best_leaf, change position, etc...)
	void select(const float cpuct, Ndarray<int, 2> board, Ndarray<int, 1> metadata);

	// expands the leaf node, backpropagates, plays the best move if necessary, resets the game if it's been terminated. Not threadsafe.
	// also resets the selected leaf to null; select must be called again to select the best leaf.
	// requires: select has been called.
	void update(const float q, Ndarray<float, 3> policy);

	// auto-auto_play: whether to automatically play the next move when num_sims_to_play is reached
	MCTS(const int num_sims_per_move, float t = 1.0, bool auto_play = true, string output = "") :
		root(new MCTSNode(WHITE)), best_leaf(nullptr), best_leaf_path(), p(),
		sims(num_sims_per_move), temperature(t), default_temp(t),
		auto_play(auto_play), moves(new Move[MAX_MOVES]), nmoves(0), leaves(MAX_MOVES, pair<Move, float>(0, 0.0f)),
		move_num(1), game_num(1), output_path_base(output)
	{
		update_output();
		best_leaf_path.reserve(200);
	}

	~MCTS() { 
		if (root != nullptr) 
			recursive_delete(*root, nullptr);
		if (moves != nullptr)
			delete[] moves;
		output.close();
	}

	MCTS& operator=(MCTS other) = delete;
	MCTS(const MCTS& other) = delete;

	// because msvc compiler sucks, we have to make this ourselves.
	MCTS(MCTS&& other) noexcept :
		root(other.root),
		best_leaf(other.best_leaf),
		moves(other.moves),
		nmoves(other.nmoves),
		best_leaf_path(std::move(other.best_leaf_path)),
		leaves(std::move(other.leaves)),
		p(std::move(other.p)),
		auto_play(other.auto_play),
		sims(other.sims),
		default_temp(other.default_temp),
		output_path_base(other.output_path_base),
		output(std::move(other.output)),
		move_num(other.move_num),
		game_num(other.game_num),
		temperature(other.temperature)
	{
		other.root = nullptr;
		other.moves = nullptr;
	}
};

