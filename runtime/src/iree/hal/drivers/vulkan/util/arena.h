// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_UTIL_ARENA_H_
#define IREE_HAL_DRIVERS_VULKAN_UTIL_ARENA_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace iree {

template <typename T>
class Span {
 public:
  Span(T* data, size_t size) noexcept : data_(data), size_(size) {}

  T* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size() == 0; }

  T& operator[](size_t i) noexcept { return *(data() + i); }

 private:
  T* data_;
  size_t size_;
};

// Arena allocator.
// Allocates memory from a cached block list grown at specified intervals.
// Individual allocations cannot be freed.
// Default constructors will be called when allocating but no destructors will
// ever be called.
//
// This should be used in places where extreme dynamic memory growth is required
// to ensure that the allocations stay close to each other in memory, are easy
// to account for, and can be released together. For example, proto or file
// parsing, per-batch write-once/read-once data buffers, etc.
//
// Usage:
//   Arena arena;
//   auto t0 = arena.Allocate<MyType>();
class Arena {
 public:
  static constexpr size_t kDefaultBlockSize = 32 * 1024;
  static constexpr size_t kBlockOverhead = sizeof(void*) + sizeof(size_t);

  Arena() : Arena(kDefaultBlockSize) {}
  explicit Arena(size_t block_size);
  ~Arena();

  // Clears all data in the arena and deallocates blocks.
  // Use Reset to avoid reallocation.
  void Clear();

  // Resets data in the arena but does not deallocate blocks.
  // Use Clear to reclaim memory.
  void Reset();

  // Block size, excluding the block header.
  // This is the largest size of any allocation that can be made of the arena.
  size_t block_size() const { return block_size_; }

  // Total number of bytes that have been allocated, excluding wasted space.
  size_t bytes_allocated() const { return bytes_allocated_; }
  // Total number of bytes as blocks allocated, including wasted space.
  // If this number is much higher than bytes_allocated the block size requires
  // tuning.
  size_t block_bytes_allocated() const { return block_bytes_allocated_; }

  // Allocates an instance of the given type and calls its constructor.
  template <typename T>
  T* Allocate() {
    void* storage = AllocateBytes(sizeof(T));
    return new (storage) T();
  }

  // Allocates an instance of the given type and calls its constructor with
  // arguments.
  template <typename T, typename... Args>
  T* Allocate(Args&&... args) {
    void* storage = AllocateBytes(sizeof(T));
    return new (storage) T(std::forward<Args>(args)...);
  }

  // Allocates an array of items and returns a span pointing to them.
  template <typename T>
  Span<T> AllocateSpan(size_t count) {
    void* storage = AllocateBytes(count * sizeof(T));
    return Span<T>(reinterpret_cast<T*>(storage), count);
  }

  // Allocates a block of raw bytes from the arena.
  // Zero-byte allocations will return nullptr.
  uint8_t* AllocateBytes(size_t length);

 private:
  // Block size contains the BlockHeader, so a 1024b block size will result in
  // 1024-sizeof(BlockHeader) usable bytes.
  size_t block_size_ = kDefaultBlockSize;
  size_t bytes_allocated_ = 0;
  size_t block_bytes_allocated_ = 0;

  // Each block in the arena contains a prefixed header that lets us link the
  // blocks together (to make freeing easier) as well as tracking current byte
  // count to let us fill gaps.
  // Immediately following the header is the actual arena data, up until the
  // block size is reached.
  struct BlockHeader {
    BlockHeader* next_block;
    size_t bytes_allocated;
  };
  static_assert(sizeof(BlockHeader) == kBlockOverhead, "Block header mismatch");

  // Singly-linked list of allocated blocks in reverse allocation order (so
  // the most recently allocated block is first).
  BlockHeader* block_list_head_ = nullptr;

  // Allocated but unused blocks.
  BlockHeader* unused_block_list_head_ = nullptr;
};

}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_UTIL_ARENA_H_
