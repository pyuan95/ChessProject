#include <vector>
#include <unordered_map>
#include <stdint.h>

class MemoryManager
{
public:
    virtual char *malloc_(uint32_t size);
    virtual char *realloc_(char *ptr, uint32_t prevsize, uint32_t newsize);
    virtual void free_(char *ptr, uint32_t size);
};

class DefaultMemoryManager : public MemoryManager
{
    inline char *malloc_(uint32_t size) { return (char *)malloc(size); }
    inline char *realloc_(char *ptr, uint32_t prevsize, uint32_t newsize) { return (char *)realloc(ptr, newsize); }
    inline void free_(char *ptr, uint32_t size) { free(ptr); }
};

class MemoryBlock : public MemoryManager
{

private:
    uint32_t starting_size;
    char *memory;
    char *frontier;
    char *wall;
    std::unordered_map<uint32_t, std::vector<uint32_t>> recycling;

public:
    // cannot hold more than starting_size * 2^32 bytes!
    MemoryBlock(uint32_t blocksize_,
                uint32_t starting_size_) : starting_size(starting_size_)
    {
        memory = (char *)malloc(blocksize_);
        frontier = memory;
        wall = memory + blocksize_;
    }

    ~MemoryBlock()
    {
        free(memory);
    }

    MemoryBlock(MemoryBlock const &other) = delete;
    MemoryBlock operator=(MemoryBlock other) = delete;

    // returns the number of bytes remaining that are available in this block
    inline uint32_t memory_until_wall() { return wall - frontier; }

    inline uint32_t count_free_memory()
    {
        uint32_t tot = 0;
        for (auto i : recycling)
            tot += i.second.size() * i.first;
        return tot + memory_until_wall();
    }

    // returns nullptr if allocation was unsuccessful,
    // and the pointer otherwise
    char *malloc_(uint32_t size);
    // requires: newsize >= prevsize
    char *realloc_(char *ptr, uint32_t prevsize, uint32_t newsize);
    void free_(char *ptr, uint32_t size);
    // resizes the underlying memory array and returns the new address
    char *resize(uint32_t size);
};

char *MemoryBlock::resize(uint32_t size)
{
    char *newmemory = (char *)realloc(memory, size);
    int diff = newmemory - memory;
    memory = newmemory;
    frontier += diff;
    wall = memory + size;

    return memory;
}

char *MemoryBlock::malloc_(uint32_t size)
{
    uint32_t cur = starting_size;
    while (cur < size)
        cur += cur;
    if (recycling.count(cur) && recycling[cur].size() > 0)
    {
        uint32_t displacement = recycling[cur].back();
        recycling[cur].pop_back();
        return memory + displacement * starting_size;
    }
    if (frontier + cur > wall)
        return nullptr;
    char *result = frontier;
    frontier += cur;
    return result;
}

char *MemoryBlock::realloc_(char *ptr, uint32_t prevsize, uint32_t newsize)
{
    uint32_t cur = starting_size;
    while (cur < prevsize)
        cur += cur;
    if (newsize <= cur)
        return ptr;
    free_(ptr, prevsize);
    return malloc_(newsize);
}

void MemoryBlock::free_(char *ptr, uint32_t size)
{
    uint32_t cur = starting_size;
    while (cur < size)
        cur += cur;
    uint32_t displacement = (ptr - memory) / starting_size;
    recycling[cur].push_back(displacement);
}