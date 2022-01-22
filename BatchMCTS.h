#pragma once
#include <vector>
#include <mutex>
#include <thread>
#include <queue>
#include <unordered_set>
#include <condition_variable>
#include <iostream>
#include <time.h>
#include "MCTS.h"

struct Sector {
	Ndarray<float, 1> q;
	Ndarray<float, 4> policy;
	int sector;

	Sector(int sector, Ndarray<float, 1> q, Ndarray<float, 4> policy) : sector(sector), q(q), policy(policy) {}
};

class BatchMCTS {
private:
	float cpuct;

	std::vector<MCTS> arr;
	int num_threads;
	int batch_size;
	int num_sectors;
	Ndarray<int, 3> boards; // (batch_size * num_sectors, 8, 8)

	int cur_sector;
	std::vector<Sector> working_sectors;
	std::mutex queue_consumer_mutex; // mutex for queue_consumer
	std::mutex update_mutex; // mutex for update method
	std::mutex select_mutex; // mutex for select method
	std::condition_variable queue_add; // notify_all called when something is added to queue
	std::condition_variable queue_remove; // notify_all called when something is removed from queue

	std::thread queue_consumer_thread;
	bool alive = true; // will make false in destructor; signals queue_consumer to terminate

	void update_sector(int sector, Ndarray<float, 1> q, Ndarray<float, 4> policy);

	void process_thread(Ndarray<float, 1> q, Ndarray<float, 4> policy, int& next, const int target, std::mutex& m);

	void queue_consumer();

	Sector& get_next_sector();

	int num_working_sectors();

public:
	void select();
	void update(Ndarray<float, 1> q, Ndarray<float, 4> policy); // (batch_size), (batch_size, rows, cols, moves_per_square)
	BatchMCTS(
		int num_sims_per_move,
		float temperature,
		bool autoplay, 
		string output, // the base name of the output file.
		int num_threads,
		int batch_size,
		int num_sectors,
		float cpuct,
		Ndarray<int, 3> boards
	)
		: num_threads(num_threads),
		batch_size(batch_size),
		num_sectors(num_sectors),
		cpuct(cpuct),
		boards(boards),
		cur_sector(0)
	{
		if (boards.getShape(0) != batch_size * num_sectors
			|| boards.getShape(1) != ROWS
			|| boards.getShape(2) != COLS) {
			throw std::runtime_error("boards must have shape (batch_size * num_sectors, 8, 8)");
		}
		this->arr.reserve(batch_size * num_sectors);
		for (int i = 0; i < batch_size * num_sectors; i++) {
			string new_output = "";
			if (!output.empty()) {
				long current_time = (long) std::chrono::system_clock::now().time_since_epoch().count();
				new_output = output + "_" + std::to_string(i) + "_" + std::to_string(current_time);
			}
			this->arr.emplace_back(num_sims_per_move, temperature, autoplay, new_output);
			this->arr.back().select(cpuct, boards[i]);
		}
		working_sectors.reserve(num_sectors);
		for (int i = 0; i < num_sectors; i++) {
			working_sectors.emplace_back(
				-1,
				Ndarray<float, 1>(
					new float[batch_size],
					new long[1]{ batch_size },
					new long[1]{ 1 }
					),
				Ndarray<float, 4>(
					new float[batch_size * ROWS * COLS * MOVES_PER_SQUARE](),
					new long[4]{ batch_size, ROWS, COLS, MOVES_PER_SQUARE },
					new long[4]{ ROWS * COLS * MOVES_PER_SQUARE, COLS * MOVES_PER_SQUARE, MOVES_PER_SQUARE, 1 }
					)
			);
		}
		queue_consumer_thread = std::thread(&BatchMCTS::queue_consumer, this);
	}

	~BatchMCTS() {
		alive = false;
		queue_add.notify_all();
		queue_consumer_thread.join();
	}
};