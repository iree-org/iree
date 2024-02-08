// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/vec_stream.h"

//===----------------------------------------------------------------------===//
// iree_io_vec_stream_t
//===----------------------------------------------------------------------===//

#define IREE_IO_VEC_BLOCK_ALIGNMENT 16
#define IREE_IO_VEC_BLOCK_MIN_SIZE 1024

// Metadata and storage for a block.
// Block sizes provided by users include the metadata so that users can pick
// bucketing allocator-friendly sizes and not end up tripping into the next
// bucket. This means each block actually stores a bit less than whatever they
// request. Use IREE_IO_VEC_BLOCK_STORAGE_CAPACITY to determine the actual
// storage capacity per block based on a block size.
typedef struct iree_io_vec_block_t {
  // Next block in the stream's block linked list.
  struct iree_io_vec_block_t* next;
  // Previous block in the stream's block linked list.
  struct iree_io_vec_block_t* prev;
  // Global offset within the stream.
  iree_io_stream_pos_t offset;
  // Capacity of the block storage in bytes.
  iree_host_size_t capacity;
  // Current length of the block storage in bytes.
  // This will be under capacity if the block is at the end of the stream.
  iree_host_size_t length;
  // Block contents of size capacity.
  iree_alignas(IREE_IO_VEC_BLOCK_ALIGNMENT) uint8_t contents[/*capacity*/];
} iree_io_vec_block_t;

#define IREE_IO_VEC_BLOCK_STORAGE_CAPACITY(block_size) \
  ((block_size)-offsetof(iree_io_vec_block_t, contents))

typedef struct iree_io_vec_stream_t {
  iree_io_stream_t base;
  iree_allocator_t host_allocator;
  // Current offset within the stream. block_pos is the block containing the
  // offset.
  iree_io_stream_pos_t offset;
  // Total length of the stream. The available capacity of all blocks allocated
  // will be greater than or equal to this.
  iree_io_stream_pos_t length;
  // Size of each block allocated in bytes.
  // Uniform block sizing prevents allocator fragmentation.
  iree_host_size_t block_size;
  // Head of the block linked list, NULL if none allocated.
  iree_io_vec_block_t* block_head;
  // Tail of the block linked list, NULL if none allocated.
  iree_io_vec_block_t* block_tail;
  // Block containing the current stream offset, NULL if none allocated.
  iree_io_vec_block_t* block_pos;
} iree_io_vec_stream_t;

static const iree_io_stream_vtable_t iree_io_vec_stream_vtable;

static iree_io_vec_stream_t* iree_io_vec_stream_cast(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  return (iree_io_vec_stream_t*)base_stream;
}

IREE_API_EXPORT iree_status_t iree_io_vec_stream_create(
    iree_io_stream_mode_t mode, iree_host_size_t block_size,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  block_size =
      iree_max(IREE_IO_VEC_BLOCK_MIN_SIZE,
               iree_host_align(block_size, IREE_IO_VEC_BLOCK_ALIGNMENT));
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)block_size);

  iree_io_vec_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &iree_io_vec_stream_vtable;
  stream->base.mode = mode;
  stream->host_allocator = host_allocator;
  stream->offset = 0;
  stream->length = 0;
  stream->block_size = block_size;
  stream->block_head = stream->block_tail = stream->block_pos = NULL;

  *out_stream = &stream->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_io_vec_stream_destroy(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  iree_allocator_t host_allocator = stream->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_vec_block_t* block = stream->block_head;
  while (block) {
    iree_io_vec_block_t* next = block->next;
    iree_allocator_free(host_allocator, block);
    block = next;
  }

  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_io_vec_stream_enumerate_blocks(
    iree_io_stream_t* base_stream, iree_io_vec_stream_callback_fn_t callback,
    void* user_data) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)stream->length);

  iree_status_t status = iree_ok_status();
  for (iree_io_vec_block_t* block = stream->block_head; block != NULL;
       block = block->next) {
    status = callback(
        user_data, iree_make_const_byte_span(block->contents, block->length));
    if (!iree_status_is_ok(status)) break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_io_stream_pos_t iree_io_vec_stream_offset(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  return stream->offset;
}

static iree_io_stream_pos_t iree_io_vec_stream_length(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  return stream->length;
}

// Asserts the block list and current offset match.
static void iree_io_vec_stream_assert_valid(iree_io_vec_stream_t* stream) {
  if (!stream->block_head) return;
  IREE_ASSERT(stream->block_pos);
  IREE_ASSERT_LE(stream->block_pos->offset, stream->offset);
  IREE_ASSERT_GE(stream->block_pos->offset + stream->block_pos->length,
                 stream->offset);
}

// Extends the stream up to the new total length.
// The current stream offset is not changed though both block_head and
// block_tail may be.
static iree_status_t iree_io_vec_stream_extend(
    iree_io_vec_stream_t* stream, iree_io_stream_pos_t new_length) {
  IREE_ASSERT_ARGUMENT(stream);
  if (!new_length) return iree_ok_status();
  if (stream->length >= new_length) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, new_length);

  // Determine how many bytes we need to allocate and then allocate blocks up
  // until we reach that new total. We'll fill the current block (if any) first
  // and that may be all we need.
  iree_io_stream_pos_t remaining_bytes = new_length - stream->length;
  if (stream->block_tail != NULL) {
    // Fill the current block first. This may satisfy the entire request and
    // we can bail early.
    iree_host_size_t block_bytes =
        iree_min(remaining_bytes,
                 stream->block_tail->capacity - stream->block_tail->length);
    stream->block_tail->length += block_bytes;
    stream->length += block_bytes;
    remaining_bytes -= block_bytes;
  }
  iree_status_t status = iree_ok_status();
  iree_host_size_t block_capacity =
      IREE_IO_VEC_BLOCK_STORAGE_CAPACITY(stream->block_size);
  while (remaining_bytes > 0) {
    // Allocate a new block.
    iree_io_vec_block_t* block = NULL;
    status = iree_allocator_malloc(stream->host_allocator, stream->block_size,
                                   (void**)&block);
    if (!iree_status_is_ok(status)) break;
    iree_host_size_t block_bytes = iree_min(remaining_bytes, block_capacity);
    block->prev = stream->block_tail;
    if (block->prev) {
      block->prev->next = block;
    }
    stream->block_tail = block;
    block->next = NULL;
    if (!stream->block_head) {
      // First block, set as head.
      stream->block_head = block;
    }
    block->offset = stream->length;
    stream->length += block_bytes;
    block->capacity = block_capacity;
    block->length = block_bytes;
    remaining_bytes -= block_bytes;
    // NOTE: iree_allocator_malloc guarantees contents are zeroed.
  }
  IREE_ASSERT_EQ(stream->length, new_length);
  if (!stream->block_pos) {
    // If we just allocated the stream then set the offset 0 block.
    stream->block_pos = stream->block_head;
  }
  iree_io_vec_stream_assert_valid(stream);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_vec_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We compute a new global offset and then navigate the list based on that. We
  // could use the seek mode as a discriminator for that instead but before
  // walking we have to handle extends on the common path anyway.
  iree_io_stream_pos_t new_offset = stream->offset;
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      new_offset = seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      new_offset = stream->offset + seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      new_offset = stream->length + seek_offset;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unrecognized seek mode %u", (uint32_t)seek_mode);
  }
  if (new_offset == stream->offset) {
    // No change fast-path.
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  } else if (new_offset < 0) {
    // Trying to seek off the beginning of the stream.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "seek %u offset %" PRIi64
                            " out of stream bounds; expected 0 <= %" PRIi64,
                            (uint32_t)seek_mode, seek_offset, new_offset);
  }

  // Extend the stream if the new offset is off the current end. This will
  // allocate new empty blocks with zeroed contents.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_vec_stream_extend(stream, new_offset));

  // If the stream is not allocated then bail (seeking to offset 0 of an empty
  // stream doesn't allocate anything).
  if (!stream->block_head) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Seek to find the block containing the new offset.
  // Since we only have a linked list we have to do a walk but the direction we
  // walk will be based on where we are starting from. We special case some
  // common cases like 0 and end to avoid walking the whole list.
  if (new_offset == 0 ||
      (stream->block_head && new_offset < stream->block_head->length)) {
    // Within the first block.
    stream->block_pos = stream->block_head;
  } else if (new_offset == stream->length ||
             (stream->block_tail && new_offset >= stream->block_tail->offset)) {
    // Within the last block.
    stream->block_pos = stream->block_tail;
  } else {
    // Somewhere in the middle of the list; walk forward or backward.
    IREE_ASSERT(stream->block_pos);
    if (new_offset < stream->offset) {
      // Seeking backward.
      iree_io_vec_block_t* block = stream->block_pos;
      for (; block && block->offset > new_offset; block = block->prev) {
      }
      stream->block_pos = block;
    } else {
      // Seeking forward.
      iree_io_vec_block_t* block = stream->block_pos;
      for (; block && block->offset + block->length < new_offset;
           block = block->next) {
      }
      stream->block_pos = block;
    }
  }
  stream->offset = new_offset;
  iree_io_vec_stream_assert_valid(stream);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_vec_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  if (out_buffer_length) *out_buffer_length = 0;
  if (buffer_capacity == 0) return iree_ok_status();
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Determine how many bytes to read based on how many bytes are available
  // in the stream from the current offset.
  iree_io_stream_pos_t remaining_length = stream->length - stream->offset;
  iree_host_size_t read_bytes = buffer_capacity;
  if (buffer_capacity > remaining_length) {
    // Access exceeds remaining length.
    if (out_buffer_length) {
      // Read-to-end; we'll return less than the full capacity.
      read_bytes = remaining_length;
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "access to range [%" PRIu64 ", %" PRIu64
                              ") (%" PRIhsz
                              " bytes) out of range; stream offset %" PRIu64
                              " and length %" PRIu64 " insufficient",
                              stream->offset, stream->offset + buffer_capacity,
                              buffer_capacity, stream->offset, stream->length);
    }
  }

  // Copy bytes from blocks for the entire read length.
  uint8_t* buffer_ptr = (uint8_t*)buffer;
  iree_host_size_t read_offset = 0;
  iree_io_stream_pos_t new_offset = stream->offset;
  iree_io_vec_block_t* block = stream->block_pos;
  iree_io_stream_pos_t block_offset = new_offset - block->offset;
  while (read_offset < read_bytes) {
    IREE_ASSERT(block);
    if (new_offset >= block->offset + block->length) {
      IREE_ASSERT(block->next,
                  "should have verified length and have a next block");
      block = block->next;
      block_offset = 0;
    }
    IREE_ASSERT(block);
    iree_host_size_t block_bytes =
        iree_min(read_bytes - read_offset, block->length);
    memcpy(buffer_ptr, &block->contents[block_offset], block_bytes);
    buffer_ptr += block_bytes;
    read_offset += block_bytes;
    new_offset += block_bytes;
  }
  stream->offset = new_offset;
  stream->block_pos = block;
  iree_io_vec_stream_assert_valid(stream);

  if (out_buffer_length) *out_buffer_length = read_bytes;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_vec_stream_write(iree_io_stream_t* base_stream,
                                              iree_host_size_t buffer_length,
                                              const void* buffer) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  if (!buffer_length) return iree_ok_status();
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Extend the stream storage up to the final size from the current position.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_vec_stream_extend(stream, stream->offset + buffer_length));

  // Copy the source buffer to the blocks.
  iree_host_size_t remaining_bytes = buffer_length;
  iree_io_vec_block_t* block = stream->block_pos;
  iree_host_size_t block_offset = stream->offset - block->offset;
  const uint8_t* buffer_ptr = (const uint8_t*)buffer;
  while (remaining_bytes > 0) {
    IREE_ASSERT(block);
    if (block_offset == block->capacity) {
      IREE_ASSERT(block->next, "should have resized and have a next block");
      block = block->next;
      block_offset = 0;
    }
    IREE_ASSERT(block);
    iree_host_size_t write_bytes =
        iree_min(block->capacity - block_offset, remaining_bytes);
    memcpy(&block->contents[block_offset], buffer_ptr, write_bytes);
    buffer_ptr += write_bytes;
    remaining_bytes -= write_bytes;
    block_offset += write_bytes;
  }

  // Update the offset and block containing it for future operations.
  stream->offset += buffer_length;
  stream->block_pos = block;
  iree_io_vec_stream_assert_valid(stream);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_vec_stream_fill_1(iree_io_vec_stream_t* stream,
                                               iree_io_stream_pos_t count,
                                               uint8_t pattern) {
  // Copy the source buffer to the blocks.
  iree_host_size_t remaining_bytes = count;
  iree_io_vec_block_t* block = stream->block_pos;
  iree_host_size_t block_offset = stream->offset - block->offset;
  while (remaining_bytes > 0) {
    IREE_ASSERT(block);
    if (block_offset == block->capacity) {
      IREE_ASSERT(block->next, "should have resized and have a next block");
      block = block->next;
      block_offset = 0;
    }
    IREE_ASSERT(block);
    iree_host_size_t write_bytes =
        iree_min(block->capacity - block_offset, remaining_bytes);
    memset(&block->contents[block_offset], pattern, write_bytes);
    remaining_bytes -= write_bytes;
    block_offset += write_bytes;
  }

  // Update the offset and block containing it for future operations.
  stream->offset += count;
  stream->block_pos = block;
  iree_io_vec_stream_assert_valid(stream);

  return iree_ok_status();
}

static iree_status_t iree_io_vec_stream_fill(iree_io_stream_t* base_stream,
                                             iree_io_stream_pos_t count,
                                             const void* pattern,
                                             iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(pattern);
  iree_io_vec_stream_t* stream = iree_io_vec_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow the stream to the entire new length (if needed).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_io_vec_stream_extend(stream,
                                stream->offset + count * pattern_length),
      "growing stream to fill bounds");

  // TODO(benvanik): efficient fill - we should be able to partition into
  // prior block and some new number of blocks. The tricky part is that the
  // alignment is 1 so we may need to split the pattern across the boundary.
  // For now we fast path pattern_length 1 and are slow for everything else.
  if (pattern_length == 1) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_vec_stream_fill_1(stream, count, *((const uint8_t*)pattern)));
  } else {
    for (iree_io_stream_pos_t i = 0; i < count; ++i) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_io_vec_stream_write(base_stream, pattern_length, pattern));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_vec_stream_map_read(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_const_byte_span_t* out_span) {
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "vec streams do not support mapping");
}

static iree_status_t iree_io_vec_stream_map_write(iree_io_stream_t* base_stream,
                                                  iree_host_size_t length,
                                                  iree_byte_span_t* out_span) {
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "vec streams do not support mapping");
}

static const iree_io_stream_vtable_t iree_io_vec_stream_vtable = {
    .destroy = iree_io_vec_stream_destroy,
    .offset = iree_io_vec_stream_offset,
    .length = iree_io_vec_stream_length,
    .seek = iree_io_vec_stream_seek,
    .read = iree_io_vec_stream_read,
    .write = iree_io_vec_stream_write,
    .fill = iree_io_vec_stream_fill,
    .map_read = iree_io_vec_stream_map_read,
    .map_write = iree_io_vec_stream_map_write,
};
