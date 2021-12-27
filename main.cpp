#include <iostream>
#include "tests.h"
#include <vector>
using namespace std;

class Big {
public:
    int x[100][100];
    vector<Big*> bigs;

    void test() {
        for (int i = 0; i < 10000; i++) {
            Big* b = new Big();
            bigs.push_back(b);
        }
    }
};

int main2()
{
    Big b;
    for (int i = 0; i < 100; i++)
    {
        b.test();
        b = *b.bigs[0];
    }

    return 0;
}

int main() {
    init_rand();
    initialise_all_databases();
	zobrist::initialise_zobrist_keys();
	run_all_tests();
	return 0;
}
