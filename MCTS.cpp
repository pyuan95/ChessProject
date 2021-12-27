#include "MCTS.h"
#include <math.h>
#include <random>
#include <ctime>
#include <fstream>

bool compare_leaf(pair<Move, float> x, pair<Move, float> y) {
	return x.second < y.second; // sort increasing
}

int invert(int piece)
{
	if (piece <= 5)
		return piece + 8;
	else if (piece <= 13)
		return piece - 8;
	return 14;
}

float rand_num(float s, float l) {
	float f = std::rand() / (RAND_MAX + EPSILON); // unif(0,1)
	f = f * (l - s) + s;
	return f;
}

// writes a position to the given board.
// 0...5 = our side (pawn, knight, bishop, rook, queen, king)
// 8...13 = their side (pawn, knight, bishop, rook, queen, king)
// 14 = empty
template <Color color>
void writePosition(const Position& p, Ndarray<int, 2>& board) {
	int piece;
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			piece = p.at(create_square(File(c), Rank(r)));
			if (color == BLACK) // invert and rotate
			{
				piece = invert(piece);
				board[ROWS - r - 1][COLS - c - 1] = piece;
			}
			else {
				board[r][c] = piece;
			}
		}
	}
}

template <Color color>
void writePosition(const Position& p, int board[ROWS][COLS]) {
	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLS; c++)
		{
			int piece = p.at(create_square(File(c), Rank(r)));
			if (color == BLACK) // invert and rotate
			{
				piece = invert(piece);
				r = ROWS - r - 1;
				c = COLS - c - 1;
			}
			board[r][c] = piece;
		}
	}
}

// returns -1 if it is a victory for black, 0 for a draw, 1 for a victory for white.
// requires: p is a terminal position (there are no legal moves)
float evaluateTerminalPosition(const Position& p)
{
	if (p.in_check<WHITE>())
		return -1.0;
	else if (p.in_check<BLACK>())
		return 1.0;
	else
		return 0.0;
}

void init_rand()
{
	std::srand(std::time(nullptr));
}

PolicyIndex move2index(const Position& p, Move m, Color color)
{
	Square f = m.from();
	Square t = m.to();
	PieceType pt = type_of(p.at(f));
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
			cout << "Illegal argument provided to move2index: queen move";
			throw runtime_error("Illegal argument provided to move2index: Queen move");
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
			cout << "Illegal argument provided to move2index: knight move";
			throw runtime_error("Illegal argument provided to move2index: Knight move");
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
		default :
			cout << "Illegal argument provided to move2index: pawn move:";
			throw runtime_error("Illegal argument provided to move2index: pawn move");
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
		cout << "Illegal piecetype" << "\n";
		cout << "Piecetype:" << pt << "\n";
		throw runtime_error("Invalid PieceType, should never happen!");
	}
	PolicyIndex policyIndex;
	if (color == BLACK)
		f = Square(63 - f);
	policyIndex.r = rank_of(f);
	policyIndex.c = file_of(f);
	policyIndex.i = i;
	return policyIndex;
}

MCTSNode* MCTSNode::select_best_child(const float cpuct)
{
	MCTSNode* res = nullptr;
	float best = -FLT_MAX;
	for (MCTSNode* child : children)
	{
		float u = cpuct * child->get_prob();
		u *= std::sqrt(get_num_times_selected()) / (1.0 + child->get_num_times_selected());
		if (u - child->get_mean_q() > best) // mean q is evaluated for the other side; we want to minimize the other side's success.
		{
			best = u - child->get_mean_q();
			res = child;
		}
		// std::cout << "Child count: " << child->get_num_times_selected() << "\tvalue: " << u - child->get_mean_q() << "\n";
	}

	if (leaves.empty() || best > cpuct * leaves.back().second * std::sqrt(get_num_times_selected()))
		return res;
	else if (!leaves.empty()) {
		MCTSNode* newNode;
		auto k = leaves.back();
		if (get_color() == WHITE)
			newNode = new MCTSNode(k.second, k.first, BLACK);
		else
			newNode = new MCTSNode(k.second, k.first, WHITE);
		leaves.pop_back();
		children.push_back(newNode);

		return newNode;
	}
	return nullptr;
}

float* MCTSNode::calculate_policy_cumsum(float temperature)
{
	float* cum_sum = new float[children.size()];
	float d = get_num_times_selected();
	*cum_sum = pow(children[0]->get_num_times_selected() / d, 1 / temperature);
	for (int i = 1; i < children.size(); i++) {
		*(cum_sum + i) = *(cum_sum + i - 1) + pow(children[i]->get_num_times_selected() / d, 1 / temperature);
	}
	return cum_sum;
}

float* MCTSNode::calculate_policy(float temperature)
{
	float* cum_sum = calculate_policy_cumsum(temperature);
	float largest = *(cum_sum + children.size() - 1) + pow(10, -10); // to avoid div by 0 errors
	float prev = 0;
	for (int i = 0; i < children.size(); i++)
	{
		*(cum_sum + i) = (*(cum_sum + i) / largest) - prev;
		prev += *(cum_sum + i);
	}
	return cum_sum;
}

vector<pair<Move, float>> MCTSNode::policy(float temperature)
{
	float* s = calculate_policy(temperature);
	vector<pair<Move, float>> policy;
	for (int i = 0; i < children.size(); i++)
		policy.push_back(pair<Move, float>(children[i]->get_move(), *(s + i)));
	delete[] s;
	for (auto k : leaves)
		policy.push_back(pair<Move, float>(k.first, 0.0));  // each leaf has a count of zero, so also a prob of zero.
	return policy;
}

MCTSNode* MCTSNode::select_best_child_by_count(float temperature)
{
	float* cum_sum = calculate_policy_cumsum(temperature);
	float thresh = rand_num(0, cum_sum[children.size() - 1] + EPSILON);
	for (int i = 0; i < children.size(); i++)
	{
		if (*(cum_sum + i) >= thresh)
		{
			delete[] cum_sum;
			return children[i];
		}
	}
	delete[] cum_sum;
	if (children.size() > 0)
		return children.back();
	else  // the only children are leaves, which have count == 0
		return nullptr;
}

float MCTSNode::minimax_evaluation()
{
	if (children.size() == 0) // we don't want to do minimax if we have no children, or all the children are leaves.
		return get_mean_q();
	else
	{
		float min_eval = DBL_MAX;
		for (MCTSNode* child : children) {
			if (!child->is_leaf() || child->is_terminal_position())
				// find the move that's worst for our opponent. use that move.
				min_eval = std::min(child->minimax_evaluation(), min_eval);
		}

		return -1 * min_eval;
	}
}

void MCTSNode::expand(Position& p, Ndarray<float, 3> policy, const Move* moves, size_t size)
{
	if (!is_terminal_position()) {
		float tot = 0;
		for (int j = 0; j < size; j++)
		{
			PolicyIndex i = move2index(p, moves[j], color);
			auto kk = policy[2];
			float prob = exp(policy[i.r][i.c][i.i]);
			tot += prob;
			leaves.push_back(pair<Move, float>(moves[j], prob));
		}
		for (auto &k : leaves)
			k.second /= tot;
		std::sort(leaves.begin(), leaves.end(), compare_leaf);
	}
}

void MCTSNode::backup(float q) {
	num_times_selected++; 
	this->q += q; 
}

void MCTS::add_move(BoardState board_state, Policy p, Move m, Color c)
{
	Game& g = game_history.back();
	g.color_history.push_back(c);
	g.move_history.push_back(m);
	g.policy_history.push_back(p);
	g.board_history.push_back(board_state);

	// std::cout << g.move_history.size() << "\n";
}

void MCTS::declare_winner(float c)
{
	game_history.back().winner = c;
}

size_t MCTS::move_num() {
	return game_history.back().board_history.size() + 1; // when there are 0 elements on the stack we are on move number 1.
}

size_t MCTSNode::size() {
	size_t tot = 1 + leaves.size();
	for (MCTSNode* child : children)
		tot += child->size();
	return tot;
}

void MCTS::serialize_games(string name)
{
	ofstream out(name);
	for (Game g : game_history)
	{
		for (Move m : g.move_history)
			out << m << " ";
		out << "\n";
	}
	out.close();
}

void MCTS::new_game()
{
	// delete and reset old resources
	if (root != nullptr) 
		delete root;
	root = new MCTSNode(1.0, Move(0), WHITE);
	best_leaf = nullptr;
	best_leaf_path.clear();
	p = Position();
	temperature = default_temp;
	game_history.push_back(Game());
}

void MCTS::select(const float cpuct, Ndarray<int, 2> board) {
	while ((root->get_num_times_selected() < sims || auto_play) && select_helper(cpuct, board)) {
		// select helper returned true. means p is at a terminal position now.
		float q = evaluateTerminalPosition(p);
		if (best_leaf->get_color() == BLACK)
			q *= -1;
		update(q, DUMMY_POLICY);
	}
}

bool MCTS::select_helper(const float cpuct, Ndarray<int, 2>& board)
{
	MCTSNode* cur = root;
	best_leaf_path.push_back(cur);
	while (!(cur->is_leaf())) {
		MCTSNode* child = cur->select_best_child(cpuct);
		if (cur->get_color() == WHITE)
			p.play<WHITE>(child->get_move());
		else if (cur->get_color() == BLACK)
			p.play<BLACK>(child->get_move());
		cur = child;
		best_leaf_path.push_back(cur);
	}
	best_leaf = cur;
	if (!best_leaf->is_terminal_position()) // check if our leaf is a terminal position. If so, mark it.
	{
		bool itp = false;
		if (best_leaf->get_color() == WHITE) {
			white_moves = new MoveList<WHITE>(p);
			if (white_moves->size() == 0) {
				delete white_moves;
				white_moves = nullptr;
				itp = true;
			}
		}
		else
		{
			black_moves = new MoveList<BLACK>(p);
			if (black_moves->size() == 0) {
				delete black_moves;
				black_moves = nullptr;
				itp = true;
			}
		}
		if (itp)
			best_leaf->mark_terminal_position();
	}
	// our leaf is a terminal position. return true.
	if (best_leaf->is_terminal_position())
	{
		return true;
	}
	else // our leaf is not a terminal position.
	{
		if (best_leaf->get_color() == WHITE) {
			writePosition<WHITE>(p, board);
		}
		else if (best_leaf->get_color() == BLACK) {
			writePosition<BLACK>(p, board);
		}
		return false;
	}
}

void MCTS::update(const float q, Ndarray<float, 3> policy)
{
	// std::cout << p;
	if (root->get_num_times_selected() >= sims && !auto_play) {
		// over the max sims and no auto-play; simply exit. but we need to maintain our invariants.
		if (best_leaf_path.size() > 0 || best_leaf != nullptr) {
			std::cout << "max sims reached and auto play off, but somehow leaf path was not empty or best_leaf non-null.";
			throw runtime_error("max sims error");
		}
		return;
	}

	Color best_leaf_color = best_leaf->get_color();
	if (white_moves != nullptr && best_leaf_color == WHITE) {
		best_leaf->expand(p, policy, white_moves->begin(), white_moves->size());
		delete white_moves;
		white_moves = nullptr;
	}
	else if (black_moves != nullptr && best_leaf_color == BLACK) {
		best_leaf->expand(p, policy, black_moves->begin(), black_moves->size());
		delete black_moves;
		black_moves = nullptr;
	}
	best_leaf = nullptr;

	// backpropagate the q value.
	MCTSNode* cur;
	while (!best_leaf_path.empty())
	{
		cur = best_leaf_path.back();
		best_leaf_path.pop_back();
		cur->backup(cur->get_color() == best_leaf_color ? q : -1.0 * q);
		if (cur != root) {
			if (cur->get_color() == WHITE)
				p.undo<BLACK>(cur->get_move());
			else
				p.undo<WHITE>(cur->get_move());
		}
	}

	// if the the root's visit count is equal to the number of sims, we need to play our move.
	while (root->get_num_times_selected() >= sims && auto_play)
	{
		// first, calculate the policy.
		Policy policy;
		vector<pair<Move, float>> policy_vec = root->policy(temperature);
		for (const auto &move_and_prob : policy_vec)
		{
			PolicyIndex pidx;
			Move move = move_and_prob.first;
			float prob = move_and_prob.second;
			if (root->get_color() == WHITE)
				pidx = move2index(p, move, WHITE);
			else
				pidx = move2index(p, move, BLACK);
			policy.p[pidx.r][pidx.c][pidx.i] = prob;
		}

		// now, write the board state.
		BoardState board_state;
		if (root->get_color() == WHITE)
			writePosition<WHITE>(p, board_state.b);
		else
			writePosition<BLACK>(p, board_state.b);

		// update the board position
		MCTSNode* best_child = root->select_best_child_by_count(temperature);
		Move m = best_child->get_move();
		Color root_color = root->get_color();
		if (root_color == WHITE)
			p.play<WHITE>(m);
		else if (root_color == BLACK)
			p.play<BLACK>(m);

		//update root
		MCTSNode* newRoot = new MCTSNode(*best_child); // make a deep copy of the best child
		delete root; // delete the contents of the old root
		root = newRoot;

		// add the move to the game.
		add_move(board_state, policy, m, root_color);

		// check to see if the game is over. if so, declare a winner and start a new game.
		root_color = root->get_color();
		if ((root_color == WHITE && MoveList<WHITE>(p).size() == 0)
			|| (root_color == BLACK && MoveList<BLACK>(p).size() == 0)) {
			declare_winner(evaluateTerminalPosition(p));
			new_game();
		}
	}
}