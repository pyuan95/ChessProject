#include "BatchMCTS.h"

Sector& BatchMCTS::get_next_sector() {
	for (int i = 0; i < num_sectors; i++) {
		int j = (i + cur_sector) % num_sectors;
		if (working_sectors[j].sector >= 0)
			return working_sectors[j];
	}
	throw std::runtime_error("should not happen!");
}

int BatchMCTS::num_working_sectors() {
	int count = 0;
	for (int i = 0; i < num_sectors; i++) {
		if (working_sectors[i].sector >= 0)
			count++;
	}
	return count;
}

void BatchMCTS::process_thread(Ndarray<float, 1> q, Ndarray<float, 4> policy, int& next, const int target, std::mutex& m) {
	while (1) {
		m.lock();
		if (next < target) {
			int cur_idx = next;
			int policy_index = next - (target - batch_size);
			next++;
			// std::cout << cur_idx << " " << policy_index << " " << std::this_thread::get_id() << "\n";
			m.unlock();

			arr[cur_idx].update(q[policy_index], policy[policy_index]);
			arr[cur_idx].select(cpuct, boards[cur_idx], metadata[cur_idx]);
		}
		else {
			m.unlock();
			break;
		}
	}
}

void BatchMCTS::queue_consumer() {
	while (alive) {
		std::unique_lock<std::mutex> lock(queue_consumer_mutex);
		while (num_working_sectors() == 0 && alive) {
			// std::cout << "waiting queue_consumer\n";
			queue_add.wait(lock);
		}
		lock.unlock();
		if (alive) {
			Sector& s = get_next_sector();
			update_sector(s.sector, s.q, s.policy);
			s.sector = -1;
			queue_remove.notify_all();
		}
	}
}

void BatchMCTS::update_sector(int sector, Ndarray<float, 1> q, Ndarray<float, 4> policy) {
	int next = sector * batch_size;
	int target = next + batch_size;
	std::mutex m;
	std::vector<std::thread> threads;
	for (int i = 0; i < num_threads; i++) {
		std::thread thd(&BatchMCTS::process_thread, this, q, policy, std::ref(next), target, std::ref(m));
		threads.push_back(std::move(thd));
	}
	for (std::thread& t : threads) {
		t.join();
	}
	// std::cout << "updated sector " << cur_sector << "!\n";
	// std::cout << "num working sectors: " << num_working_sectors() << "\n";
}

void BatchMCTS::select() {
	std::unique_lock<std::mutex> lock(select_mutex);
	// wait until the current sector is out of the working sectors.
	// then select has been called and all is good in the hood.
	// std::cout << cur_sector << "\n";
	while (working_sectors[cur_sector].sector >= 0) {
		// std::cout << "waiting select...";
		queue_remove.wait(lock);
	}
	// std::cout << "selected sector " << cur_sector << "!\n";
}

void BatchMCTS::update(Ndarray<float, 1> q, Ndarray<float, 4> policy) {
	queue_consumer_mutex.lock();
	working_sectors[cur_sector].sector = cur_sector;
	working_sectors[cur_sector].q = q;
	working_sectors[cur_sector].policy = policy;
	cur_sector = (cur_sector + 1) % num_sectors;
	queue_add.notify_all(); // notify queue_consumer
	queue_consumer_mutex.unlock();
}