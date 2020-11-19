#include "iree/hal/host/pre_allocator.h"

#include <stdio.h>
#include <stdlib.h>

#include <cstring>

static uint allocation_size = 1 << 30;

static char* data_ = (char*)malloc(allocation_size);
static int64_t offset_ = 0;
static bool mem_init_ = false;

void* PreAllocator::malloc(unsigned int n_bytes) {
  if (!mem_init_) {
    memset(data_, 0, allocation_size);
    mem_init_ = true;
  }
  void* address = (void*)(data_ + offset_);
  offset_ += n_bytes;
  return address;
}
void PreAllocator::free_allocation(unsigned int n_bytes) { offset_ -= n_bytes; }

void PreAllocator::release_memory() {
  printf("release_memory of size: %d\n", allocation_size);
  free(data_);
}
