#include <iostream>
#include "ndarray.h"
#include "BatchMCTS.h"
#include "tablebase_evaluation.h"
extern "C"
{
    namespace
    {
        int myfunc(numpyArray<float> array1, numpyArray<float> array2, char *test, int x)
        {
            Ndarray<float, 3> a(array1);
            Ndarray<float, 3> b(array2);
            std::cout << "inside my func: " << a << "\n\n\n\n";
            double sum = 0.0;

            for (int i = 0; i < a.getShape(0); i++)
            {
                for (int j = 0; j < a.getShape(1); j++)
                {
                    for (int k = 0; k < a.getShape(2); k++)
                    {
                        a[i][j][k] = 2.0 * b[i][j][k];
                        sum += a[i][j][k];
                    }
                }
            }
            std::cout << "after my func: " << a << "\n\n\n\n";
            std::cout << "test! " << test << "\n";
            std::cout << "x! " << x << "\n";
            return sum;
        }

        void initialize(char *tablebase_path)
        {
            init_rand();
            initialise_all_databases();
            zobrist::initialise_zobrist_keys();
            init_move2index_cache();
            init_tablebase(tablebase_path);
        }

        void deleteBatchMCTS(BatchMCTS *m)
        {
            delete m;
        }

        BatchMCTS *createBatchMCTS(int num_sims_per_move,
                                   float temperature,
                                   bool autoplay,
                                   char *output,
                                   int num_threads,
                                   int batch_size,
                                   int num_sectors,
                                   float cpuct,
                                   numpyArray<int> boards_,
                                   numpyArray<int> metadata_)
        {
            Ndarray<int, 3> boards(boards_);
            Ndarray<int, 2> metadata(metadata_);
            BatchMCTS *m = new BatchMCTS(num_sims_per_move,
                                         temperature,
                                         autoplay,
                                         output,
                                         num_threads,
                                         batch_size,
                                         num_sectors,
                                         cpuct,
                                         boards,
                                         metadata);
            return m;
        }

        void select(BatchMCTS *m)
        {
            m->select();
        }

        void update(BatchMCTS *m, numpyArray<float> q_, numpyArray<float> policy_)
        {
            Ndarray<float, 1> q(q_);
            Ndarray<float, 4> policy(policy_);
            m->update(q, policy);
        }

        void set_temperature(BatchMCTS *m, float temp)
        {
            m->set_temperature(temp);
        }

        void play_best_moves(BatchMCTS *m)
        {
            m->play_best_moves();
        }

        bool all_games_over(BatchMCTS *m)
        {
            return m->all_games_over();
        }

        double proportion_of_games_over(BatchMCTS *m)
        {
            return m->proportion_of_games_over();
        }

        void results(BatchMCTS *m, numpyArray<int> res_)
        {
            Ndarray<int, 1> res(res_);
            m->results(res);
        }

        int current_sector(BatchMCTS *m)
        {
            return m->current_sector();
        }
    }
} // end extern "C"
