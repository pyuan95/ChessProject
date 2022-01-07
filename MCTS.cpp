#include "MCTS.h"

bool compare_leaf(pair<Move, float> x, pair<Move, float> y) {
	return x.second > y.second; // sort decreasing by probability
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
	std::srand((unsigned int) std::time(nullptr));
}

void move2index(const Position& p, Move m, Color color, PolicyIndex& policyIndex)
{
	Square f = m.from();
	PieceType pt = type_of(p.at(f));
	if (color == BLACK)
		f = Square(63 - f);
	policyIndex.r = rank_of(f);
	policyIndex.c = file_of(f);
	policyIndex.i = move2index_cache[static_cast<int>(color)][static_cast<int>(pt)][m.get_representation()];
}

MCTSNode* MCTSNode::add_leaf() {
	reallocate_memory();

	get_node_at(num_expanded) = MCTSNode(~get_color());
	num_expanded++;

	return begin_nodes() + num_expanded - 1;
}

std::pair<MCTSNode*, Move> MCTSNode::select_best_child(const float cpuct)
{
	MCTSNode* res = nullptr;
	Move best_move(0);
	float best(-FLT_MAX);
	float start(cpuct * std::sqrt((float) get_num_times_selected()));
	float u;
	if (num_expanded < num_children) {
		best = start * get_prob_at(num_expanded); // calculating with the leaf w the highest prob
		best_move = get_move_at(num_expanded);
	}

	for (int i = 0; i < num_expanded; i++) {
		// mean q is evaluated for the other side; we want to minimize the other side's success.
		MCTSNode& child = get_node_at(i);
		u = start * get_prob_at(i) / (1.0f + child.get_num_times_selected()) - child.get_mean_q();
		if (u > best) {
			best = u;
			res = &child;
			best_move = get_move_at(i);
		}
		// std::cout << "Child count: " << child.get_num_times_selected() << "\tvalue: " << u - child.get_mean_q() << "\n";
	}

	if (res == nullptr && num_expanded < num_children) {
		// the best leaf won!
		res = add_leaf();
	}

	return std::pair<MCTSNode*, Move>(res, best_move);
}

float* MCTSNode::calculate_policy_cumsum(float temperature)
{
	float* cum_sum = new float[num_expanded];
	float d = (float) get_num_times_selected();
	*cum_sum = pow(get_node_at(0).get_num_times_selected() / d, 1 / temperature);
	for (int i = 1; i < num_expanded; i++) {
		cum_sum[i] = cum_sum[i - 1] + pow(get_node_at(i).get_num_times_selected() / d, 1 / temperature);
	}
	return cum_sum;
}

float* MCTSNode::calculate_policy(float temperature)
{
	float* cum_sum = calculate_policy_cumsum(temperature);
	float largest = cum_sum[num_expanded - 1] + EPSILON; // to avoid div by 0 errors
	float prev = 0.0f;
	for (int i = 0; i < num_expanded; i++) {
		cum_sum[i] = (cum_sum[i] / largest) - prev;
		prev += cum_sum[i];
	}
	return cum_sum;
}

vector<pair<Move, float>> MCTSNode::policy(float temperature)
{
	float* s = calculate_policy(temperature);
	vector<pair<Move, float>> policy;
	for (int i = 0; i < num_expanded; i++)
		policy.push_back(pair<Move, float>(get_move_at(i), s[i]));
	delete[] s;
	return policy;
}

std::pair<MCTSNode*, Move> MCTSNode::select_best_child_by_count(float temperature)
{
	if (num_expanded == 0) {
		return std::pair<MCTSNode*, Move>(nullptr, 0);
	}
	float* cum_sum = calculate_policy_cumsum(temperature);
	float thresh = rand_num(0, cum_sum[num_expanded - 1] + EPSILON);
	for (int i = 0; i < num_expanded; i++) {
		if (cum_sum[i] >= thresh) {
			delete[] cum_sum;
			return std::pair<MCTSNode*, Move>(begin_nodes() + i, get_move_at(i));
		}
	}
	// should never happen, but to be safe...
	delete cum_sum;
	return std::pair<MCTSNode*, Move>(begin_nodes() + num_expanded - 1, get_move_at(num_expanded - 1));
}

float MCTSNode::minimax_evaluation()
{
	if (num_expanded == 0) // we don't want to do minimax if we have no children, or all the children are leaves.
		return get_mean_q();
	else
	{
		float min_eval = FLT_MAX;
		for (int i = 0; i < num_expanded; i++) {
			MCTSNode& child = get_node_at(i);
			if (!child.is_leaf() || child.is_terminal_position())
				// find the move that's worst for our opponent. use that move.
				min_eval = std::min(child.minimax_evaluation(), min_eval);
		}

		return -1 * min_eval;
	}
}

void MCTSNode::expand(Position& p, Ndarray<float, 3> policy, const Move* moves, size_t size)
{
	if (!is_terminal_position() && is_leaf() && size > 0) {
		// first, initialize the memory for children
		num_children = (uint8_t) size;
		init_memory();

		float tot = 0;
		vector<pair<Move, float>> leaves(size, pair<Move, float>(0, 0.0f));
		PolicyIndex policyIndex;
		for (int i = 0; i < size; i++) {
			move2index(p, moves[i], get_color(), policyIndex);
			float prob = exp(policy[policyIndex.r][policyIndex.c][policyIndex.i]);
			tot += prob;
			leaves[i].first = moves[i];
			leaves[i].second = prob;
		}
		std::sort(leaves.begin(), leaves.end(), compare_leaf);

		for (int i = 0; i < num_children; i++) {
			set_move_at(i, leaves[i].first);
			set_prob_at(i, leaves[i].second / tot);
		}
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
	size_t tot = 1 + num_children - num_expanded;
	for (int i = 0; i < num_expanded; i++)
		tot += get_node_at(i).size();
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
	root = new MCTSNode(WHITE);
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
	best_leaf_path.push_back(std::pair<MCTSNode*, Move>(cur, 0));
	while (!(cur->is_leaf())) {
		auto child = cur->select_best_child(cpuct);
		if (cur->get_color() == WHITE)
			p.play<WHITE>(child.second);
		else
			p.play<BLACK>(child.second);
		best_leaf_path.push_back(std::pair<MCTSNode*, Move>(child.first, child.second));

		cur = child.first;
	}
	best_leaf = cur;
	if (!best_leaf->is_terminal_position()) {
		// check if our leaf is a terminal position. If so, mark it.
		bool itp(false);
		if (best_leaf->get_color() == WHITE) {
			white_moves = new MoveList<WHITE>(p);
			if (white_moves->size() == 0) {
				delete white_moves;
				white_moves = nullptr;
				itp = true;
			}
		}
		else {
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
	if (best_leaf->is_terminal_position()) {
		return true;
	}
	// our leaf is not a terminal position.
	else {
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
	Move m;
	while (!best_leaf_path.empty()) {
		cur = best_leaf_path.back().first;
		m = best_leaf_path.back().second;
		best_leaf_path.pop_back();
		cur->backup(cur->get_color() == best_leaf_color ? q : -1.0f * q);
		if (cur != root) {
			if (cur->get_color() == WHITE)
				p.undo<BLACK>(m);
			else
				p.undo<WHITE>(m);
		}
	}

	// if the the root's visit count is equal to the number of sims, we need to play our move.
	while (root->get_num_times_selected() >= sims && auto_play)
	{
		// first, calculate the policy.
		Policy policy;
		vector<pair<Move, float>> policy_vec = root->policy(temperature);
		PolicyIndex pidx;
		for (const auto &move_and_prob : policy_vec)
		{
			Move move = move_and_prob.first;
			float prob = move_and_prob.second;
			if (root->get_color() == WHITE)
				move2index(p, move, WHITE, pidx);
			else
				move2index(p, move, BLACK, pidx);
			policy.p[pidx.r][pidx.c][pidx.i] = prob;
		}

		// now, write the board state.
		BoardState board_state;
		if (root->get_color() == WHITE)
			writePosition<WHITE>(p, board_state.b);
		else
			writePosition<BLACK>(p, board_state.b);

		// update the board position
		pair<MCTSNode*, Move> best_child = root->select_best_child_by_count(temperature);
		Move m = best_child.second;
		Color root_color = root->get_color();
		if (root_color == WHITE)
			p.play<WHITE>(m);
		else
			p.play<BLACK>(m);

		MCTSNode* newRoot = new MCTSNode(*(best_child.first)); // shallow copy the best child
		recursive_delete(*root, best_child.first);
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

void recursive_delete(MCTSNode& n, MCTSNode* ignore) {
	if (&n == ignore)
		return;
	MCTSNode* nodes = n.begin_nodes();
	for (int i = 0; i < n.get_num_expanded(); i++) {
		recursive_delete(nodes[i], ignore);
	}
	if (n.get_num_children() > 0)
		free(n.begin_children());
}