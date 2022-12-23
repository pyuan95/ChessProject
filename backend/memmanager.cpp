#include "memmanager.h"
#include <iostream>
int64_t MemoryBlock::resize(uint32_t size)
{
    std::cout << "resizing from " << this->size() << " bytes to " << size << " bytes. had " << count_free_memory() << " bytes free before\n";
    uint8_t *newmemory = (uint8_t *)realloc(memory, size);
    int64_t diff = newmemory - memory;
    memory = newmemory;
    frontier += diff; // newmemory + (frontier - memory)
    wall = memory + size;

    return diff;
}

uint8_t *MemoryBlock::malloc_(uint32_t size)
{
    uint32_t cur = get_piece_size(size);
    if (recycling.count(cur) && recycling[cur].size() > 0)
    {
        uint32_t displacement = recycling[cur].back();
        recycling[cur].pop_back();
        return memory + displacement * starting_allocation_size;
    }
    if (frontier + cur > wall)
        return nullptr;

    uint8_t *result = frontier;
    frontier += cur;
    // std::cout << "malloc" << (long)result << "\t" << size << "\n";
    return result;
}

uint8_t *MemoryBlock::realloc_(uint8_t *ptr, uint32_t prevsize, uint32_t newsize)
{
    uint32_t cur = get_piece_size(prevsize);
    if (newsize <= cur)
        return ptr;
    uint8_t *newptr = malloc_(newsize);
    memcpy(newptr, ptr, std::min(prevsize, newsize));
    free_(ptr, prevsize);
    return newptr;
}

void MemoryBlock::free_(uint8_t *ptr, uint32_t size)
{
    uint32_t cur = get_piece_size(size);
    uint32_t displacement = (ptr - memory) / starting_allocation_size;
    recycling[cur].push_back(displacement);
}