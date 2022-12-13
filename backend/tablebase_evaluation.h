
#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tbprobe.h"
#include <string>
#include <iostream>

#define BOARD_RANK_1 0x00000000000000FFull
#define BOARD_FILE_A 0x8080808080808080ull
#define square(r, f) (8 * (r) + (f))
#define rank(s) ((s) >> 3)
#define file(s) ((s)&0x07)
#define board(s) ((uint64_t)1 << (s))

static const char *wdl_to_str[5] =
    {
        "0-1",
        "1/2-1/2",
        "1/2-1/2",
        "1/2-1/2",
        "1-0"};

struct pos
{
    uint64_t white;
    uint64_t black;
    uint64_t kings;
    uint64_t queens;
    uint64_t rooks;
    uint64_t bishops;
    uint64_t knights;
    uint64_t pawns;
    uint8_t castling;
    uint8_t rule50;
    uint8_t ep;
    bool turn;
    uint16_t move;
};

/*
 * Parse a FEN string.
 */
static bool parse_FEN(struct pos *pos, const char *fen);
inline void init_tablebase(const char *path)
{
    assert(tb_init(path));
    assert(TB_LARGEST > 0);
}

// returns non-zero if ok, zero if failure
// assigns -1 to value for black win, 0 for draw, 1 for white win
int probe(int *value, std::string fen);

/*
int main()
{
    std::string path = "./tablebase/";
    init_tablebase(path.c_str());
    int value;
    std::cout << probe(&value, "8/8/5k2/6p1/8/3K4/8/8 w - - 0 1") << "\t" << value << "\n";

    for (int i = 0; i < 100000; i++)
    {
        probe(&value, "8/8/5k2/6p1/8/3K4/8/8 w - - 0 1");
    }
    return 0;
}
*/
