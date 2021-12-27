#pragma once
#include <vector>
#include "Constants.h"
#include "position.h"
#include "types.h"
#include "tables.h"
#include "ndarray.h"

using namespace std;

// stores the indices in the policy array
struct PolicyIndex {
	int r;
	int c;
	int i;
};

// wrapper class for the policy array; we can't directly store arrays in a vector.
struct Policy {
	float p[ROWS][COLS][MOVES_PER_SQUARE] = { 0 };
};

// wrapper class for the board state; we can't directly store arrays in a vector.
struct BoardState {
	int b[ROWS][COLS];
};

// each MCTS object will contain a member of vector<Game>, which it will use to store completed games.
struct Game {
	vector<Policy> policy_history;  // policy[i] is the policy for board_history[i]
	vector<BoardState> board_history; // stores the board_history. boards are rotated and pieces are inverted if necessary.
	vector<Move> move_history; // move_history[i] is the move from board_history[i] to board_history[i + 1]
	vector<Color> color_history; // color_history[i] is the color of the turn to move for board_history[i].
	float winner = 2; // -1 for black, 0 for draw, 1 for white, 2 for in progress.
};

// initializes the random seed with the current time.
void init_rand();

// including these in the header files for testing purposes.

template<Color color>
void writePosition(const Position& p, Ndarray<int, 2>& board);

template<Color color>
void writePosition(const Position& p, int board[ROWS][COLS]);

PolicyIndex move2index(const Position& p, Move m, Color color);

/*
The class that represents a node in the MCTS Tree
*/
class MCTSNode {
private:
	const float prob;
	float q;
	Move move;
	uint32_t num_times_selected;
	bool itp;
	const Color color;
	vector<MCTSNode*> children; // may or may not be leaves.
	vector<pair<Move, float>> leaves; // to save memory, we dont make MCTSNode from leaf until it's about to be expanded.

public:
	 
	// returns true if this node is a leaf. node may or may not be terminal as well.
	inline bool is_leaf() { return children.size() == 0 && leaves.size() == 0; }

	// returns the color of the side whose turn it is to play
	inline Color get_color() { return color; }

	// returns true if this node is a terminal position ( no moves available, game has ended. is not a leaf. )
	inline bool is_terminal_position() { return itp; }

	// marks the current node as terminal position.
	inline void mark_terminal_position() { itp = true; }

	// returns the probability of selecting this node.
	inline float get_prob() { return prob; }

	// returns the number of times this node was selected.
	inline int get_num_times_selected() { return num_times_selected; }

	// returns the move that was used to get from the previous position to the current position.
	inline Move get_move() { return move; }

	// returns the Q value for this node.
	inline float get_mean_q() { return num_times_selected > 0 ? q / num_times_selected : 0.0; }

	// the q value must be calculated given the color of the node.
	inline void backup(float q);

	// expands the node given the policy. the q value must be updated using the update method.
	// requires: the current node is a leaf. If it is terminal, no changes are made.
	// policy is appropriately rotated if the player is black.
	inline void expand(Position& p, Ndarray<float, 3> policy, const Move* moves, size_t size);

	// returns the number of nodes currently under this tree.
	size_t size();

	// returns the minimax evaluation of the tree rooted at this node.
	float minimax_evaluation();

	// calculates the policy and returns it as a cumulative sum array. res[i] == total sum of children 0....i inclusive.
	// size == children.size()
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
	MCTSNode* select_best_child(const float cpuct);

	// returns the child to play based on visit count with the given temperature parameter.
	// if there are no children that are not leaves, nullptr is returned.
	// this can happen when the current node is a leaf (could be terminal position), or if
	// this node was never selected for expansion.
	MCTSNode* select_best_child_by_count(float temperature=1.0);

	MCTSNode(const float probability, Move m, Color c) : prob(probability), q(0), num_times_selected(0),
		itp(false), color(c), children(), move(m) {}
	MCTSNode(const MCTSNode& other) : prob(other.prob), q(other.q), num_times_selected(other.num_times_selected),
		itp(other.itp), color(other.color), move(other.move), children() {
		for (MCTSNode* c : other.children)
		{
			children.push_back(new MCTSNode(*c));
		}
	}
	MCTSNode& operator=(MCTSNode other) = delete; // we have constant members so this operator cannot be implemented.
	~MCTSNode() {
		for (MCTSNode* c : children) {
			delete c;
		}
	}
};

/*
Represents an MCTS Tree
CLass invariant: root -> color == p.turn().
*/
class MCTS {
private:
	MCTSNode* root;
	MCTSNode* best_leaf;
	MoveList<WHITE>* white_moves;
	MoveList<BLACK>* black_moves;
	vector<MCTSNode*> best_leaf_path;
	Position p;
	vector<Game> game_history;
	bool auto_play;
	const int sims;
	const float default_temp;

	// adds a move to the current game
	// we want to pass by value bc board_state, p, m, c, are made on stack.
	void add_move(BoardState board_state, Policy p, Move m, Color c);

	// declares a winner for the current game
	void declare_winner(float c);

	// adds a new game and resets all PIVs.
	void new_game();

	// the select helper. returns true if select-helper led to selecting on a terminal position and nothing was written.
	// false otherwise (even if autoplay off and max sims reached)
	// white_moves and black_moves will be set to nullptr if this function returns false.
	// requires: max sims not reached.
	bool select_helper(const float cpuct, Ndarray<int, 2>& board);
	
public:

	// the temperature to use for calculating the policy and selecting the best child by count.
	float temperature;

	inline Position position() { return p; } // for debugging only

	inline void set_position(Position pos) { 
		this->p = pos; 
		if (root != nullptr)
			delete root;
		if (pos.turn() == WHITE)
			root = new MCTSNode(1.0, Move(0), WHITE);
		else
			root = new MCTSNode(1.0, Move(0), BLACK);
		best_leaf = nullptr;
		best_leaf_path.clear();
	}

	// returns whether we have reached the sim limit. relevent only if auto_play is false.
	inline bool reached_sim_limit() { return root->get_num_times_selected() >= sims; }

	// returns the current number of sims in the root.
	inline int current_sims() { return root->get_num_times_selected(); }

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
	inline Move get_best_move(float temperature) { return root->select_best_child_by_count(temperature)->get_move(); }

	// returns the number of nodes in this tree.
	inline size_t size() { return root->size(); }

	// returns the game history, including the game currently in progress.
	inline const vector<Game>& games() { return game_history; }

	// save the game data into a file with the given prefix
	void serialize_games(string name);

	// the move number that the current game is on.
	// may be used to set the temperature (in the paper, tmeperature goes from 1 -> close to 0 after move 30).
	size_t move_num();

	// selects the best leaf thru MCTS and writes the position and the legal moves. Not threadsafe.
	// Additionally, sets the best_leaf* to point to the selected node.
	// if max sims is reached and auto play is off, this method does nothing (does not assign best_leaf, change position, etc...)
	void select(const float cpuct, Ndarray<int, 2> board);

	// expands the leaf node, backpropagates, plays the best move if necessary, resets the game if it's been terminated. Not threadsafe.
	// also resets the selected leaf to null; select must be called again to select the best leaf.
	// requires: select has been called.
	void update(const float q, Ndarray<float, 3> policy);

	// auto-auto_play: whether to automatically play the next move when num_sims_to_play is reached
	MCTS(const int num_sims_per_move, float t=1.0, bool auto_play=true) :
		root(new MCTSNode(1.0, Move(0), WHITE)), best_leaf(nullptr), best_leaf_path(), p(),
		sims(num_sims_per_move), game_history(1, Game()), temperature(t), default_temp(t),
		auto_play(auto_play), white_moves(nullptr), black_moves(nullptr) {}

	~MCTS() { if (root != nullptr) delete root; }

	MCTS& operator=(MCTS other) = delete;
	MCTS(const MCTS& other) = delete;
};

