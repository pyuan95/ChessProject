#pragma once

#include <vector>
#include <unordered_map>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iostream>
class MemoryManager
{
public:
    virtual uint8_t *malloc_(uint32_t size) = 0;
    virtual uint8_t *realloc_(uint8_t *ptr, uint32_t prevsize, uint32_t newsize) = 0;
    virtual void free_(uint8_t *ptr, uint32_t size) = 0;
    virtual uint32_t memory_until_wall() = 0;
    // TODO: size() may exceed 4GB in the future. consider fixing.
    virtual uint32_t size() = 0;
    virtual int64_t resize(uint32_t size) = 0;
    virtual void reset() = 0;
};

class DefaultMemoryManager : public MemoryManager
{
    inline uint8_t *malloc_(uint32_t size) { return (uint8_t *)malloc(size); }
    inline uint8_t *realloc_(uint8_t *ptr, uint32_t prevsize, uint32_t newsize) { return (uint8_t *)realloc(ptr, newsize); }
    inline void free_(uint8_t *ptr, uint32_t size) { free(ptr); }
    inline uint32_t memory_until_wall() { return (uint32_t)4e9; }
    inline uint32_t size() { return (uint32_t)4e9; }
    inline int64_t resize(uint32_t size) { return 0; }
    inline void reset() {}
};

class MemoryBlock : public MemoryManager
{

private:
    static const uint32_t precalculate = 2048;
    uint32_t const starting_allocation_size;
    uint32_t const starting_blocksize;
    uint8_t *memory;
    uint8_t *frontier;
    uint8_t *wall;
    std::unordered_map<uint32_t, std::vector<uint32_t>> recycling;
    std::vector<uint32_t> memotable;

    inline uint32_t get_piece_size(uint32_t size)
    {
        if (size < precalculate)
            return memotable[size];
        uint32_t cur = starting_allocation_size;
        while (cur < size)
            cur += cur;
        return cur;
    }

public:
    // cannot hold more than starting_allocation_size * 2^32 bytes!
    MemoryBlock(uint32_t const blocksize_,
                uint32_t const starting_size_) : starting_allocation_size(starting_size_),
                                                 starting_blocksize(blocksize_),
                                                 memotable(precalculate, 0)
    {
        memory = (uint8_t *)malloc(blocksize_);
        frontier = memory;
        wall = memory + blocksize_;
        uint32_t cur = starting_allocation_size;
        for (int i = 0; i < precalculate; i++)
        {
            while (i > cur)
            {
                cur += cur;
            }
            memotable[i] = cur;
        }
    }

    ~MemoryBlock()
    {
        free(memory);
    }

    MemoryBlock(MemoryBlock &&other) : starting_blocksize(other.starting_blocksize), starting_allocation_size(other.starting_allocation_size)
    {
        recycling = std::move(other.recycling);
        std::swap(memory, other.memory);
        std::swap(frontier, other.frontier);
        std::swap(wall, other.wall);
    }

    MemoryBlock(MemoryBlock const &other) = delete;
    MemoryBlock &operator=(MemoryBlock other) = delete;

    // returns the number of bytes remaining that are available in this block
    inline uint32_t memory_until_wall()
    {
        if (frontier < wall)
            return wall - frontier;
        else
            return 0;
    }

    inline uint32_t size() { return wall - memory; }

    inline uint32_t count_free_memory()
    {
        uint32_t tot = 0;
        for (auto i : recycling)
            tot += i.second.size() * i.first;
        return tot + memory_until_wall();
    }

    inline void reset()
    {
        memory = (uint8_t *)realloc(memory, starting_blocksize);
        frontier = memory;
        wall = memory + starting_blocksize;
        recycling.clear();
    }

    // returns nullptr if allocation was unsuccessful,
    // and the pointer otherwise
    uint8_t *malloc_(uint32_t size);
    uint8_t *realloc_(uint8_t *ptr, uint32_t prevsize, uint32_t newsize);
    void free_(uint8_t *ptr, uint32_t size);
    // resizes the underlying memory array and returns new_address - old_address
    // add the return value to each pointer allocated to get new valid pointer
    int64_t resize(uint32_t size);
};