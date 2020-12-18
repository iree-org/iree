// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>

#include "experimental/remoting/iree/remoting/support/platform.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"

#ifndef IREE_REMOTING_SUPPORT_IO_BUFFER_H_
#define IREE_REMOTING_SUPPORT_IO_BUFFER_H_

namespace iree {
namespace remoting {

class IoBuffer;
class IoBufferPool;

//===----------------------------------------------------------------------===//
// IO Buffers
//===----------------------------------------------------------------------===//

// Buffer within a IoBufferPool.
// The IoBuffer itself is a POD type that does not require destruction.
// It is physically laid out with the actual buffer data trailing it:
//
//   - IoBuffer (sizeof IoBuffer)
//   - padding to alignof(IoBuffer)
//   - char buffer_data[buffer_size]
//   - padding to alignof(IoBuffer)
//
// While the alignment is not strictly necessary around the data, it simplifies
// size planning when placing consecutive buffers in an allocation.
//
// Channel buffers are reference counted, allowing the same physical backing
// buffer to be lifetime extended across IO operations in various ways. Eac
// IoBuffer::Ptr instance strongly holds on to one reference. Additional
// references can be made by calling Dup().
//
// Thread safety: Not currently thread safe (including reference counting).
class IoBuffer {
  struct PoolDeleter {
    void operator()(IoBuffer *buffer) {
      if (--buffer->ref_count_ == 0) {
        buffer->Recycle();
      }
    }
  };

 public:
  using Ptr = std::unique_ptr<IoBuffer, PoolDeleter>;

  size_t size() const;
  void *data();
  std::uint8_t *data_bytes() { return static_cast<std::uint8_t *>(data()); }

  // Creates a new strong reference to the same backing data.
  Ptr Dup() {
    ref_count_ += 1;
    return Ptr(this);
  }

 private:
  IoBuffer(IoBufferPool *owner, IoBuffer *next) : next_(next), owner_(owner) {}
  void Recycle();
  int ref_count_ = 0;
  IoBuffer *next_;
  IoBufferPool *owner_;
  friend class IoBufferPool;
  friend class IoBufferVec;
};

static_assert(std::is_standard_layout<IoBuffer>::value,
              "IoBuffer should be standard_layout");
#if __cplusplus >= 201703L
static_assert(std::is_trivially_destructible<IoBuffer>::value,
              "IoBuffer should be trivially_destructible");
#endif

// A pool of same-sized buffers intended for use in processing IO streams.
// Thread safety: Not currently thread safe (including reference counting).
class IoBufferPool {
 public:
  static constexpr size_t kDefaultSlabBufferCount = 8;
  IoBufferPool(size_t buffer_data_size,
               size_t slab_buffer_count = kDefaultSlabBufferCount)
      : buffer_data_size_(buffer_data_size),
        slab_buffer_count_(slab_buffer_count) {
    GrowBuffers();
  }
  ~IoBufferPool();

  // Gets a IoBuffer from the free-list, extending it as necessary.
  IoBuffer::Ptr Get() {
    if (!head_) GrowBuffers();
    IoBuffer *current = head_;
    assert(current->ref_count_ == 0);
    current->ref_count_ = 1;
    head_ = current->next_;
    return IoBuffer::Ptr(current);
  }

  //===--------------------------------------------------------------------===//
  // Memory layout
  //===--------------------------------------------------------------------===//
  size_t buffer_data_size() const { return buffer_data_size_; }

  // The size in memory of a buffer, consisting of the IoBuffer struct,
  // internal alignment, data buffer, and trailing alignment.
  size_t buffer_physical_size() const {
    constexpr size_t align = alignof(IoBuffer);
    size_t accum = AlignTo(sizeof(IoBuffer), align);
    accum += buffer_data_size();
    accum = AlignTo(accum, align);
    return accum;
  }

  // Byte offset relative to a IoBuffer pointer of the data.
  static constexpr size_t buffer_data_offset() {
    return AlignTo(sizeof(IoBuffer), alignof(IoBuffer));
  }

 private:
  // Buffers are allocated in a contiguous slab of memory, which is tracked
  // in a linked list for cleanup.
  struct Slab {
    ~Slab();
    void *data;
    std::unique_ptr<Slab> next;
  };

  // Aligns |value| up to the given power-of-two |alignment| if required.
  // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
  static constexpr size_t AlignTo(size_t value, size_t alignment) {
    return (value + (alignment - 1)) & ~(alignment - 1);
  }

  // Allocates a new slab.
  void GrowBuffers();

  size_t buffer_data_size_;
  size_t slab_buffer_count_;

  IoBuffer *head_ = nullptr;
  std::unique_ptr<Slab> slab_head_;
  friend class IoBuffer;
};

inline size_t IoBuffer::size() const { return owner_->buffer_data_size(); }
inline void *IoBuffer::data() {
  void *base = static_cast<void *>(this);
  char *base_bytes = static_cast<char *>(base);
  return static_cast<void *>(base_bytes + IoBufferPool::buffer_data_offset());
}

inline void IoBuffer::Recycle() {
  IREE_DLOG(INFO) << "  - Recycle buffer to pool: "
                  << static_cast<void *>(this);
  next_ = owner_->head_;
  owner_->head_ = this;
}

//===----------------------------------------------------------------------===//
// IoBufferVec
//===----------------------------------------------------------------------===//

// A wrapper around platform-specific iovec-like structs (base pointer + len).
// This also adds the capability to attach an owned IoBuffer to back each
// entry. In this way, buffer references can ber carried across various
// boundaries, and the references will be released when this instance is
// cleared, destroyed or returned to its pool.
class IoBufferVec {
  struct PoolDeleter {
    void operator()(IoBufferVec *v);
  };

 public:
  class Pool;
  using Ptr = std::unique_ptr<IoBufferVec, PoolDeleter>;

  IoBufferVec() { link_.owner = nullptr; }
  IoBufferVec(const IoBufferVec &) = delete;
  IoBufferVec(const IoBufferVec &&) = delete;
  IoBufferVec &operator=(const IoBufferVec &) = delete;
  ~IoBufferVec();

  // Clears all entries.
  void clear();
  void reserve(uint32_t capacity) { Grow(capacity); }
  uint32_t size() const { return size_; }

  // Flattens the first |len| bytes of this iovec into |dest|. If there are
  // less than |len| bytes, the contents of |dest| are undefined.
  void FlattenTo(void *dest, size_t len);

  // Copies up to |total_bytes| from this IoBufferVec into |other|, consuming
  // data from any entries available. Returns |bytes_transferred|
  // indicating how many bytes were transferred to |other|.
  int32_t TransferTo(IoBufferVec &other, int32_t max_bytes);

  const portable_iovec_t *front() {
    assert(size_ > 0 && "IoBufferVec is empty");
    return entries_;
  }

  const portable_iovec_t *operator[](uint32_t index) {
    assert(index < size_ && "IoBufferVec index out of bounds");
    return entries_ + index;
  }

  // Direct access to the iov_base.
  void *iov_base(uint32_t index) {
    return IREE_REMOTING_IOVEC_BASE(*(*this)[index]);
  }
  uint8_t *iov_base_bytes(uint32_t index) {
    return static_cast<uint8_t *>(iov_base(index));
  }
  // Direct access to the iov_len.
  size_t iov_len(uint32_t index) {
    return IREE_REMOTING_IOVEC_LEN(*(*this)[index]);
  }

  // Gets the total size of all iovec entries.
  // Meant for debugging: This loops, so don't rely on it for hot path code
  // unless intended.
  size_t total_iov_len() {
    size_t accum = 0;
    for (uint32_t i = 0, e = size(); i < e; ++i) {
      accum += iov_len(i);
    }
    return accum;
  }

  // Backing buffer for the |index|'th entry or nullptr if not backed by a
  // buffer.
  IoBuffer *buffer(uint32_t index) {
    return extras_[index].backing_buffer.get();
  }

  // Adds an iovec entry for the given raw pointer/length pair, potentially
  // backed by a buffer.
  void add(void *base, size_t len, IoBuffer::Ptr buffer = IoBuffer::Ptr()) {
    size_t index = size_;
    size_ += 1;
    if (size_ >= capacity_) Grow(0);
    IREE_REMOTING_IOVEC_BASE(entries_[index]) =
        IREE_REMOTING_IOVEC_FROM_PTR(base);
    IREE_REMOTING_IOVEC_LEN(entries_[index]) = len;
    new (extras_ + index) Extra();
    extras_[index].backing_buffer = std::move(buffer);
  }

  // Appends the contents of |other| onto this one, duping buffers as needed.
  void append(IoBufferVec &other) {
    uint32_t this_size = size();
    uint32_t other_size = other.size();
    uint32_t new_size = this_size + other_size;
    if (capacity_ < new_size) Grow(new_size);
    std::memcpy(entries_ + this_size, other.entries_,
                sizeof(entries_[0]) * other_size);
    std::memcpy(extras_ + this_size, other.extras_,
                sizeof(extras_[0]) * other_size);
    for (uint32_t i = 0, e = other_size; i < e; ++i) {
      IoBuffer *backing_buffer = other.buffer(i);
      if (backing_buffer) backing_buffer->ref_count_ += 1;
    }
    size_ = new_size;
  }

 private:
  // Note: Order of members is laid out to minimize the size.
  union {
    // If is_free_ == 1, then the instance is in the free-list and the next
    // pointer refers to the next free instance in the list.
    IoBufferVec *next;
    // If is_free_ == 0, then the instance is in use and owned by the given
    // pool (if owner != nullptr). If owner == nullptr in this case, the
    // instance is not owned (normal allocation).
    IoBufferVec::Pool *owner;
  } link_;

  // |capacity| sized contiguous array of portable_iovec_t, matching the
  // platform specific struct for passing to vector IO routines.
  portable_iovec_t *entries_ = nullptr;
  // |capacity| sized array of extra information for each portable iovec
  // entry. This is maintained separated because we want to preserve the
  // exact binary, contiguous layout of the iovec structure themselves, since
  // that can be passed losslessly to platform-specific kernel calls.
  struct Extra {
    IoBuffer::Ptr backing_buffer;
  };
  Extra *extras_ = nullptr;

  // Grows the contents. Guaranteed to increase the capacity by at least 1
  // and promote contents to a heap buffer. If |new_capacity| > 0, then sizes
  // it to the maximum of the current capacity or the stated capacity.
  void Grow(uint32_t new_capacity);

  uint32_t capacity_ : 15 = 0;
  uint32_t size_ : 15 = 0;
  uint32_t is_free_ : 1 = 0;
  uint32_t is_inline_ : 1 = 1;
};

// A pool of IoBufferVec, typically used in contexts where regular usage
// patterns are expected to stabilize to a fixed set of instances, saving
// continual re-allocation.
class IoBufferVec::Pool {
 public:
  ~Pool();

  // Gets a pooled IoBufferVec from the free-list, extending it as necessary.
  IoBufferVec::Ptr Get() {
    IoBufferVec *current = head_;
    if (current) {
      // From the free-list.
      assert(current->is_free_);
      assert(current->size_ == 0);
      head_ = current->link_.next;
      current->is_free_ = 0;
      current->link_.owner = this;
      return IoBufferVec::Ptr(current);
    }

    // Create new.
    IREE_DLOG(INFO) << "Allocate new IoBufferVec (for pool)";
    current = new IoBufferVec();
    current->link_.owner = this;
    assert(current->is_free_ == 0);
    return IoBufferVec::Ptr(current);
  }

 private:
  IoBufferVec *head_ = nullptr;
  friend class IoBufferVec::PoolDeleter;
};

inline void IoBufferVec::PoolDeleter::operator()(IoBufferVec *v) {
  v->clear();
  assert(v->is_free_ == 0);
  assert(v->link_.owner != nullptr);
  // Note that owner and next are aliases in the link_ union. Order below
  // matters.
  IoBufferVec::Pool *owner = v->link_.owner;
  v->is_free_ = 1;
  v->link_.next = owner->head_;
  owner->head_ = v;
}

inline void IoBufferVec::FlattenTo(void *dest, size_t len) {
  for (uint32_t i = 0, e = size(); i < e; ++i) {
    const void *src = iov_base(i);
    size_t chunk_len = std::min(iov_len(i), len);
    std::memcpy(dest, src, chunk_len);
    len -= chunk_len;
    if (chunk_len == 0) break;
  }
}

inline int32_t IoBufferVec::TransferTo(IoBufferVec &other, int32_t max_bytes) {
  // TODO: This function is doing too much and should be refactored/simplified.
  assert(max_bytes >= 0);
  int32_t remain = max_bytes;
  int32_t retired_count = 0;
  for (uint32_t i = 0, e = size(); i < e; ++i) {
    if (remain == 0) break;
    auto &cur_iov_base_ptr = IREE_REMOTING_IOVEC_BASE(entries_[i]);
    auto &cur_iov_len = IREE_REMOTING_IOVEC_LEN(entries_[i]);
    auto &cur_backing_buffer = extras_[i].backing_buffer;
    int32_t consume_len = std::min(remain, static_cast<int32_t>(cur_iov_len));
    IoBuffer::Ptr dupped_buffer =
        cur_backing_buffer ? cur_backing_buffer->Dup() : IoBuffer::Ptr();
    // Consume from a raw iovec.
    other.add(iov_base(i), consume_len, std::move(dupped_buffer));
    cur_iov_len -= consume_len;
    cur_iov_base_ptr = IREE_REMOTING_IOVEC_FROM_PTR(
        reinterpret_cast<uint8_t *>(cur_iov_base_ptr) + consume_len);
    if (cur_iov_len == 0) {
      cur_backing_buffer.reset();
      retired_count += 1;
    }
    remain -= consume_len;
  }

  // Compact.
  if (retired_count > 0) {
    uint32_t new_size = size() - retired_count;
    std::memmove(entries_, entries_ + retired_count,
                 sizeof(entries_[0]) * new_size);
    std::memmove(extras_, extras_ + retired_count,
                 sizeof(extras_[0]) * new_size);
    size_ = new_size;
  }

  return max_bytes - remain;
}

}  // namespace remoting
}  // namespace iree

#endif  // IREE_REMOTING_SUPPORT_IO_BUFFER_H_
