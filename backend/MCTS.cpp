#include "MCTS.h"

bool compare_leaf(pair<Move, float> x, pair<Move, float> y) {
	return x.second > y.second; // sort decreasing by probability
}

float rand_num(float s, float l) {
	float f = std::rand() / (RAND_MAX + EPSILON); // unif(0,1)
	f = f * (l - s) + s;
	return f;
}

// array must be initialized to all zeros
template<Color color>
void writeLegalMoves(Position& p, LegalMoves& legal_moves) {
	MoveList<color> l(p);
	PolicyIndex i;
	for (Move m : l) {
		move2index(p, m, color, i);
		legal_moves.l[i.r][i.c][i.i] = 1;
	}
}

// returns -1 if it is a victory for black, 1 for a victory for white, 0 otherwise
float evaluateTerminalPosition(const Position& p)
{
	if (p.in_check<WHITE>())
		return -1.0f;
	else if (p.in_check<BLACK>())
		return 1.0f;
	else
		return 0.0f;
}

void init_rand() {
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

void MCTSNode::select_best_child(const float cpuct, std::pair<MCTSNode*, Move>& child)
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
	child.first = res;
	child.second = best_move;
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
		policy.emplace_back(get_move_at(i), s[i]);
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

float MCTSNode::minimax_evaluation() {
	if (num_expanded == 0) // we don't want to do minimax if we have no children, or all the children are leaves.
		return get_mean_q();
	else {
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

void MCTSNode::expand(
	Position& p,
	Ndarray<float, 3>& policy,
	const Move* moves,
	size_t size,
	vector<pair<Move, float>>& leaves
)
{
	if (!is_terminal_position() && is_leaf() && size > 0) {
		// first, initialize the memory for children
		num_children = (uint8_t) size;
		init_memory();

		float tot = 0;
		PolicyIndex policyIndex;
		for (int i = 0; i < size; i++) {
			move2index(p, moves[i], get_color(), policyIndex);
			float prob = exp(policy[policyIndex.r][policyIndex.c][policyIndex.i]);
			tot += prob;
			leaves[i].first = moves[i];
			leaves[i].second = prob;
		}

		std::sort(leaves.begin(), leaves.begin() + size, compare_leaf);

		for (int i = 0; i < size; i++) {
			set_move_at(i, leaves[i].first);
			set_prob_at(i, leaves[i].second);
		}
	}
}

void MCTSNode::backup(float q) {
	num_times_selected++; 
	this->q += q; 
}

void MCTS::add_move(BoardState& board_state, Policy& policy, LegalMoves& legal_moves, Move& m, Color& c) {
	if (output.is_open()) {
		output << board_state << "\n";
		for (int r = 0; r < ROWS; r++) {
			for (int c = 0; c < COLS; c++) {
				for (int i = 0; i < MOVES_PER_SQUARE; i++) {
					if (legal_moves.l[r][c][i]) {
						output << r << "," << c << "," << i << "," << policy.p[r][c][i] << ",";
					}
				}
			}
		}
		output << "\n";
		output << m << "\n";
		output << c << "\n";
	}
	move_num++;
}

void MCTS::declare_winner(float c) {
	long res = std::lround(c);
	if (output.is_open()) {
		output << res << " WINNER!\n";
		output.flush();
	}
}

void MCTS::new_game() {
	// delete and reset old resources
	if (root != nullptr)
		delete root;
	root = new MCTSNode(WHITE);
	best_leaf = nullptr;
	best_leaf_path.clear();
	best_leaf_path.reserve(200);
	p = Position();
	temperature = default_temp;
	game_num++;
	move_num = 1;
	update_output();
}

int MCTS::move_number() {
	return move_num;
}

int MCTS::game_number() {
	return game_num;
}

size_t MCTSNode::size() {
	size_t tot = 1 + num_children - num_expanded;
	for (int i = 0; i < num_expanded; i++)
		tot += get_node_at(i).size();
	return tot;
}

bool MCTS::select(const float cpuct, Ndarray<int, 2> board, Ndarray<int, 1> metadata) {
	return select_helper(cpuct, board, metadata);
}

bool MCTS::select_helper(const float cpuct, Ndarray<int, 2>& board, Ndarray<int, 1>& metadata) {
	MCTSNode* cur = root;
	best_leaf_path.emplace_back(cur, 0);
	std::pair<MCTSNode*, Move> child(0, 0);
	while (!(cur->is_leaf())) {
		cur->select_best_child(cpuct, child);
		if (cur->get_color() == WHITE)
			p.play<WHITE>(child.second);
		else
			p.play<BLACK>(child.second);
		best_leaf_path.emplace_back(child.first, child.second);

		cur = child.first;
	}
	best_leaf = cur;
	if (!best_leaf->is_terminal_position()) {
		// check if our leaf is a terminal position. If so, mark it.
		bool itp(false);
		if (best_leaf->get_color() == WHITE) {
			Move* last = p.generate_legals<WHITE>(moves);
			nmoves = last - moves;
			if (nmoves == 0) {
				itp = true;
			}
		}
		else {
			Move* last = p.generate_legals<BLACK>(moves);
			nmoves = last - moves;
			if (nmoves == 0) {
				itp = true;
			}
		}
		if (itp)
			best_leaf->mark_terminal_position();
	}

	if (best_leaf->get_color() == WHITE) {
		writePosition<WHITE>(p, board, metadata);
	}
	else if (best_leaf->get_color() == BLACK) {
		writePosition<BLACK>(p, board, metadata);
	}
	return best_leaf->is_terminal_position();
}

void MCTS::play_best_move() {
	// first, calculate the policy.
	if (root->is_terminal_position()) {
		// only can happen when autoplay disabled
		return;
	}
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

	// write the legal moves;
	LegalMoves legal_moves;
	if (root->get_color() == WHITE)
		writeLegalMoves<WHITE>(p, legal_moves);
	else
		writeLegalMoves<BLACK>(p, legal_moves);

	// now, write the board state.
	BoardState board_state;
	if (root->get_color() == WHITE)
		writePosition<WHITE>(p, board_state.b, board_state.m);
	else
		writePosition<BLACK>(p, board_state.b, board_state.m);

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
	add_move(board_state, policy, legal_moves, m, root_color);

	if (auto_play && move_number() == 40) {
		temperature = 0.25;
	}

	// check to see if the game is over. if so, declare a winner and start a new game.
	if (root->is_terminal_position() && auto_play) {
		declare_winner(evaluateTerminalPosition(p));
		new_game();
	}
}

void MCTS::update(const float q, Ndarray<float, 3> policy)
{
	// if auto-play is true we will never be over the sim limit
	// but let's sanity check this just in case.
	if (root->get_num_times_selected() >= sim_limit && auto_play) {
		std::string err = "somehow went over max-sim_limit with auto-play enabled. should be impossible!";
		std::cout << err << "\n";
		throw runtime_error(err);
	}

	// if auto-play is disabled and we reach the sim limit,
	// then simply unwind the invariants and return
	if (root->get_num_times_selected() >= sim_limit && !auto_play) {
		best_leaf = nullptr;
		Move m;
		MCTSNode* cur;
		while (!best_leaf_path.empty()) {
			cur = best_leaf_path.back().first;
			m = best_leaf_path.back().second;
			best_leaf_path.pop_back();
			if (cur != root) {
				if (cur->get_color() == WHITE)
					p.undo<BLACK>(m);
				else
					p.undo<WHITE>(m);
			}
		}
		return;
	}

	float val = q;
	if (best_leaf->is_terminal_position()) {
		// p is at terminal position
		val = evaluateTerminalPosition(p);
		if (best_leaf->get_color() == BLACK)
	 		val *= -1;
	}

	Color best_leaf_color = best_leaf->get_color();
	if (nmoves > 0 && best_leaf_color == WHITE) {
		best_leaf->expand(p, policy, moves, nmoves, leaves);
	}
	else if (nmoves > 0 && best_leaf_color == BLACK) {
		best_leaf->expand(p, policy, moves, nmoves, leaves);
	}

	best_leaf = nullptr;

	// backpropagate the q value.
	MCTSNode* cur;
	Move m;
	while (!best_leaf_path.empty()) {
		cur = best_leaf_path.back().first;
		m = best_leaf_path.back().second;
		best_leaf_path.pop_back();
		cur->backup(cur->get_color() == best_leaf_color ? val : -1.0f * val);
		if (cur != root) {
			if (cur->get_color() == WHITE)
				p.undo<BLACK>(m);
			else
				p.undo<WHITE>(m);
		}
	}

	// if the the root's visit count is equal to the number of sim_limit, we need to play our move.
	while (root->get_num_times_selected() >= sim_limit && auto_play)
		play_best_move();
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

ostream& operator<<(ostream& os, const Policy& p) {
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			for (int k = 0; k < MOVES_PER_SQUARE; k++) {
				os << std::to_string(p.p[i][j][k]);
				os << ",";
			}
		}
	}
	return os;
}

ostream& operator<<(ostream& os, const LegalMoves& l) {
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			for (int k = 0; k < MOVES_PER_SQUARE; k++) {
				os << std::to_string(l.l[i][j][k]);
				os << ",";
			}
		}
	}
	return os;
}

ostream& operator<<(ostream& os, const BoardState& b) {
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			os << std::to_string(b.b[i][j]);
			os << ",";
		}
	}
	for (int i = 0; i < METADATA_LENGTH; i++) {
		os << b.m[i] << ",";
	}
	return os;
}
