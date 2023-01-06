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

class SimpleBatchMCTS
{
private:
    float cpuct;

    int num_threads;
    int batch_size;
    Ndarray<int, 3> boards;   // (batch_size * num_sectors, 8, 8)
    Ndarray<int, 2> metadata; // (batch_size * num_sectors, 5)

    std::vector<MCTS> arr;

    inline void process_thread(Ndarray<float, 1> q, Ndarray<float, 4> policy, int start, int end)
    {
        for (int cur = start; cur < end; cur++)
        {
            arr[cur].update(q[cur], policy[cur]);
            arr[cur].select(cpuct, boards[cur], metadata[cur]);
        }
    }

public:
    // calling this function ensures that the Ndarray corresponding to the current sector has finished being selected and updated
    inline void select() {}

    // does a batch update to the current sector.
    // Requires: the underlying data of q and policy does not get destroyed within the next [num_sector] calls to update()
    // (batch_size), (batch_size, rows, cols, moves_per_square)
    inline void update(Ndarray<float, 1> q, Ndarray<float, 4> policy)
    {
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++)
        {
            int start = (int)((1.0 * i / num_threads) * batch_size);
            int end = (int)(1.0 * (i + 1) / num_threads * batch_size);
            if (i == num_threads - 1)
                end = batch_size;
            std::thread thd(&SimpleBatchMCTS::process_thread, this, q, policy, start, end);
            threads.push_back(std::move(thd));
        }

        for (std::thread &t : threads)
            t.join();
    }

    // sets the temperature for each game
    inline void set_temperature(float const temp)
    {
        for (MCTS &m : arr)
            m.temperature = temp;
    }

    inline void play_best_moves(bool reset)
    {
        for (int i = 0; i < batch_size; i++)
        {
            MCTS &m = arr[i];
            m.undo_select();
            if (reset)
                m.play_best_move_and_reset();
            else
                m.play_best_move();
            m.select(cpuct, boards[i], metadata[i]);
        }
    }

    inline bool all_games_over()
    {
        for (MCTS &m : arr)
        {
            if (!m.isover())
                return false;
        }
        return true;
    }

    inline double proportion_of_games_over()
    {
        size_t cnt = 0;
        for (MCTS &m : arr)
            cnt += (int)m.isover();

        return cnt / ((double)arr.size());
    }

    // writes the results of the games to the given array
    inline void results(Ndarray<int, 1> res)
    {
        size_t x = 0;
        for (MCTS &m : arr)
        {
            res[x] = m.terminal_evaluation();
            x++;
        }
    }

    inline int current_sector()
    {
        return 0; // for legacy
    }

    // for testing
    inline std::vector<int> sim_counts()
    {
        std::vector<int> res;
        for (MCTS &m : arr)
            res.push_back(m.current_sims());
        return res;
    }

    SimpleBatchMCTS(
        int num_sims_per_move,
        float temperature,
        bool autoplay,
        string output, // the base name of the output file.
        int num_threads,
        int batch_size,
        float cpuct,
        Ndarray<int, 3> boards,
        Ndarray<int, 2> metadata) : num_threads(num_threads),
                                    batch_size(batch_size),
                                    cpuct(cpuct),
                                    boards(boards),
                                    metadata(metadata)
    {
        if (boards.getShape(0) != batch_size || boards.getShape(1) != ROWS || boards.getShape(2) != COLS)
            throw std::runtime_error("boards must have shape (batch_size * num_sectors, 8, 8)");
        else if (metadata.getShape(0) != batch_size || metadata.getShape(1) != METADATA_LENGTH)
            throw std::runtime_error("metadata must have shape (batch_size * num_sectors, 5)");
        this->arr.reserve(batch_size);
        for (int i = 0; i < batch_size; i++)
        {
            string new_output = "";
            if (!output.empty())
            {
                long current_time = (long)std::chrono::system_clock::now().time_since_epoch().count();
                new_output = output + "_" + std::to_string(i) + "_" + std::to_string(current_time);
            }
            this->arr.emplace_back(num_sims_per_move, temperature, autoplay, new_output);
            this->arr.back().select(cpuct, boards[i], metadata[i]);
        }
    }
};