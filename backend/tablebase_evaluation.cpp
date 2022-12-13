#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tbprobe.h"
#include "tablebase_evaluation.h"
#include <string>
#include <iostream>

static bool parse_FEN(struct pos *pos, const char *fen)
{
    uint64_t white = 0, black = 0;
    uint64_t kings, queens, rooks, bishops, knights, pawns;
    kings = queens = rooks = bishops = knights = pawns = 0;
    bool turn;
    unsigned rule50 = 0, move = 1;
    unsigned ep = 0;
    unsigned castling = 0;
    char c;
    int r, f;

    if (fen == NULL)
        goto fen_parse_error;

    for (r = 7; r >= 0; r--)
    {
        for (f = 0; f <= 7; f++)
        {
            unsigned s = (r * 8) + f;
            uint64_t b = board(s);
            c = *fen++;
            switch (c)
            {
            case 'k':
                kings |= b;
                black |= b;
                continue;
            case 'K':
                kings |= b;
                white |= b;
                continue;
            case 'q':
                queens |= b;
                black |= b;
                continue;
            case 'Q':
                queens |= b;
                white |= b;
                continue;
            case 'r':
                rooks |= b;
                black |= b;
                continue;
            case 'R':
                rooks |= b;
                white |= b;
                continue;
            case 'b':
                bishops |= b;
                black |= b;
                continue;
            case 'B':
                bishops |= b;
                white |= b;
                continue;
            case 'n':
                knights |= b;
                black |= b;
                continue;
            case 'N':
                knights |= b;
                white |= b;
                continue;
            case 'p':
                pawns |= b;
                black |= b;
                continue;
            case 'P':
                pawns |= b;
                white |= b;
                continue;
            default:
                break;
            }
            if (c >= '1' && c <= '8')
            {
                unsigned jmp = (unsigned)c - '0';
                f += jmp - 1;
                continue;
            }
            goto fen_parse_error;
        }
        if (r == 0)
            break;
        c = *fen++;
        if (c != '/')
            goto fen_parse_error;
    }
    c = *fen++;
    if (c != ' ')
        goto fen_parse_error;
    c = *fen++;
    if (c != 'w' && c != 'b')
        goto fen_parse_error;
    turn = (c == 'w');
    c = *fen++;
    if (c != ' ')
        goto fen_parse_error;
    c = *fen++;
    if (c != '-')
    {
        do
        {
            switch (c)
            {
            case 'K':
                castling |= TB_CASTLING_K;
                break;
            case 'Q':
                castling |= TB_CASTLING_Q;
                break;
            case 'k':
                castling |= TB_CASTLING_k;
                break;
            case 'q':
                castling |= TB_CASTLING_q;
                break;
            default:
                goto fen_parse_error;
            }
            c = *fen++;
        } while (c != ' ');
        fen--;
    }
    c = *fen++;
    if (c != ' ')
        goto fen_parse_error;
    c = *fen++;
    if (c >= 'a' && c <= 'h')
    {
        unsigned file = c - 'a';
        c = *fen++;
        if (c != '3' && c != '6')
            goto fen_parse_error;
        unsigned rank = c - '1';
        ep = square(rank, file);
        if (rank == 2 && turn)
            goto fen_parse_error;
        if (rank == 5 && !turn)
            goto fen_parse_error;
        if (rank == 2 && ((tb_pawn_attacks(ep, true) & (black & pawns)) == 0))
            ep = 0;
        if (rank == 5 && ((tb_pawn_attacks(ep, false) & (white & pawns)) == 0))
            ep = 0;
    }
    else if (c != '-')
        goto fen_parse_error;
    c = *fen++;
    if (c != ' ')
        goto fen_parse_error;
    char clk[4];
    clk[0] = *fen++;
    if (clk[0] < '0' || clk[0] > '9')
        goto fen_parse_error;
    clk[1] = *fen++;
    if (clk[1] != ' ')
    {
        if (clk[1] < '0' || clk[1] > '9')
            goto fen_parse_error;
        clk[2] = *fen++;
        if (clk[2] != ' ')
        {
            if (clk[2] < '0' || clk[2] > '9')
                goto fen_parse_error;
            c = *fen++;
            if (c != ' ')
                goto fen_parse_error;
            clk[3] = '\0';
        }
        else
            clk[2] = '\0';
    }
    else
        clk[1] = '\0';
    rule50 = atoi(clk);
    move = atoi(fen);

    pos->white = white;
    pos->black = black;
    pos->kings = kings;
    pos->queens = queens;
    pos->rooks = rooks;
    pos->bishops = bishops;
    pos->knights = knights;
    pos->pawns = pawns;
    pos->castling = castling;
    pos->rule50 = rule50;
    pos->ep = ep;
    pos->turn = turn;
    pos->move = move;
    return true;

fen_parse_error:
    return false;
}

int probe(int *value, std::string fen)
{
    struct pos pos0;
    struct pos *pos = &pos0;
    if (!parse_FEN(pos, fen.c_str()))
    {
        // std::cout << "couldn't parse fen";
        return 0;
    }
    if (tb_pop_count(pos->white | pos->black) > TB_LARGEST)
    {
        // std::cout << "too many pieces";
        return 0;
    }

    unsigned results[TB_MAX_MOVES];
    unsigned res = tb_probe_root(pos->white, pos->black, pos->kings,
                                 pos->queens, pos->rooks, pos->bishops, pos->knights, pos->pawns,
                                 pos->rule50, pos->castling, pos->ep, pos->turn, results);
    if (res == TB_RESULT_FAILED)
    {
        // std::cout << "fetching failed";
        return 0;
    }

    // output: wdl = 0 is black win, wdl = 4 is white win, wdl = 1, 2, 3 is draw
    unsigned wdl = TB_GET_WDL(res);
    if (wdl == 0)
        *value = -1;
    else if (wdl == 4)
        *value = 1;
    else
        *value = 0;
    return 1;
}