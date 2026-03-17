// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for frame accumulator: tests resilience against arbitrary byte
// streams, various chunk sizes, and boundary conditions.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/net/channel/util/frame_accumulator.h"

// Minimum input size: 1 byte for config + at least some data.
#define MIN_INPUT_SIZE 2

// Maximum frame size to test (to avoid OOM).
#define MAX_TEST_FRAME_SIZE (64 * 1024)

// 4-byte little-endian length-prefixed protocol (same as tests).
// Returns 0 if we need more data to determine length.
// Returns the total frame size (including 4-byte header) once known.
// Invalid frame sizes (< 4 or > max sane size) are still returned and will
// be rejected by the accumulator with RESOURCE_EXHAUSTED.
static iree_host_size_t fuzz_frame_length(void* user_data,
                                          iree_const_byte_span_t available) {
  if (available.data_length < 4) {
    return 0;  // Need more data to read length field.
  }
  uint32_t frame_size = available.data[0] | (available.data[1] << 8) |
                        (available.data[2] << 16) | (available.data[3] << 24);
  // Frame size of 0 in wire format means "invalid/empty" - treat as need-more.
  // Valid frames have at least 4 bytes (the header itself).
  if (frame_size < 4) {
    // Invalid frame - return a large value that will trigger RESOURCE_EXHAUSTED
    // rather than returning 0 (which would cause infinite loop trying to get
    // more data for an already-invalid frame).
    return SIZE_MAX;
  }
  return (iree_host_size_t)frame_size;
}

// Frame complete callback - just counts frames.
static iree_status_t fuzz_on_frame_complete(void* user_data,
                                            iree_const_byte_span_t frame,
                                            iree_async_buffer_lease_t* lease) {
  uint32_t* frame_count = (uint32_t*)user_data;
  ++(*frame_count);
  return iree_ok_status();
}

// Mock lease with release tracking for leak detection.
typedef struct fuzz_mock_lease_t {
  uint8_t* data;
  size_t size;
  int release_count;  // Tracks how many times release was called.
  iree_async_buffer_lease_t lease;
} fuzz_mock_lease_t;

static void fuzz_mock_release(void* user_data,
                              iree_async_buffer_index_t buffer_index) {
  fuzz_mock_lease_t* mock = (fuzz_mock_lease_t*)user_data;
  ++mock->release_count;
}

static void fuzz_mock_lease_init(fuzz_mock_lease_t* mock, uint8_t* data,
                                 size_t size) {
  mock->data = data;
  mock->size = size;
  mock->release_count = 0;
  memset(&mock->lease, 0, sizeof(mock->lease));
  mock->lease.span = iree_async_span_from_ptr(data, size);
  mock->lease.release.fn = fuzz_mock_release;
  mock->lease.release.user_data = mock;
  mock->lease.buffer_index = 0;
}

// Verifies the lease was released exactly once (leak and double-free
// detection).
static void fuzz_verify_release(fuzz_mock_lease_t* mock) {
  if (mock->release_count != 1) {
    // release_count == 0: leak (release never called)
    // release_count > 1: double-free (release called multiple times)
    abort();
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < MIN_INPUT_SIZE) {
    return 0;
  }

  // Use first byte for configuration.
  uint8_t config = data[0];
  const uint8_t* stream_data = data + 1;
  size_t stream_size = size - 1;

  // Config bits:
  // - bits 0-2: max_frame_size = 64 << (config & 0x7) (64 to 8192 bytes)
  // - bits 3-5: chunk_size_shift for chunked pushing
  // - bit 6: whether to use chunked pushing
  // - bit 7: reserved

  iree_host_size_t max_frame_size = (iree_host_size_t)64 << (config & 0x7);
  if (max_frame_size > MAX_TEST_FRAME_SIZE) {
    max_frame_size = MAX_TEST_FRAME_SIZE;
  }

  bool use_chunks = (config & 0x40) != 0;
  iree_host_size_t base_chunk_size = (iree_host_size_t)1
                                     << ((config >> 3) & 0x7);

  // Allocate accumulator storage.
  iree_host_size_t storage_size =
      iree_net_frame_accumulator_storage_size(max_frame_size);
  void* storage = malloc(storage_size);
  if (!storage) {
    return 0;
  }

  uint32_t frame_count = 0;
  iree_net_frame_accumulator_t* accumulator =
      (iree_net_frame_accumulator_t*)storage;
  iree_net_frame_length_callback_t frame_length_callback = {fuzz_frame_length,
                                                            NULL};
  iree_net_frame_complete_callback_t on_frame_complete = {
      fuzz_on_frame_complete, &frame_count};
  iree_status_t status = iree_net_frame_accumulator_initialize(
      accumulator, max_frame_size, frame_length_callback, on_frame_complete);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    free(storage);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Test 1: Push entire stream at once
  //===--------------------------------------------------------------------===//

  if (!use_chunks && stream_size > 0) {
    // Make a mutable copy since lease needs non-const pointer.
    uint8_t* stream_copy = (uint8_t*)malloc(stream_size);
    if (stream_copy) {
      memcpy(stream_copy, stream_data, stream_size);

      fuzz_mock_lease_t mock;
      fuzz_mock_lease_init(&mock, stream_copy, stream_size);

      status = iree_net_frame_accumulator_push_lease(accumulator, &mock.lease,
                                                     stream_size);
      iree_status_ignore(status);
      fuzz_verify_release(&mock);

      free(stream_copy);
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Push in variable-sized chunks (guided by fuzz input)
  //===--------------------------------------------------------------------===//

  if (use_chunks && stream_size > 0) {
    // Reset for fresh test.
    iree_net_frame_accumulator_reset(accumulator);

    // Make a mutable copy.
    uint8_t* stream_copy = (uint8_t*)malloc(stream_size);
    if (stream_copy) {
      memcpy(stream_copy, stream_data, stream_size);

      size_t offset = 0;
      size_t chunk_index = 0;
      while (offset < stream_size) {
        // Vary chunk size based on position in stream.
        size_t chunk_size = base_chunk_size;
        if (chunk_index < stream_size) {
          // Use stream bytes to vary chunk sizes for more coverage.
          chunk_size = 1 + (stream_data[chunk_index % stream_size] % 256);
        }
        if (offset + chunk_size > stream_size) {
          chunk_size = stream_size - offset;
        }

        fuzz_mock_lease_t mock;
        fuzz_mock_lease_init(&mock, stream_copy + offset, chunk_size);

        status = iree_net_frame_accumulator_push_lease(accumulator, &mock.lease,
                                                       chunk_size);
        fuzz_verify_release(&mock);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }

        offset += chunk_size;
        ++chunk_index;
      }

      free(stream_copy);
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Single-byte push (stress test buffering logic)
  //===--------------------------------------------------------------------===//

  // Only do single-byte test for small inputs to avoid O(n^2) behavior.
  if (stream_size > 0 && stream_size <= 64) {
    // Reset for fresh test.
    iree_net_frame_accumulator_reset(accumulator);

    // Make a mutable copy.
    uint8_t* stream_copy = (uint8_t*)malloc(stream_size);
    if (stream_copy) {
      memcpy(stream_copy, stream_data, stream_size);

      for (size_t i = 0; i < stream_size; ++i) {
        fuzz_mock_lease_t mock;
        fuzz_mock_lease_init(&mock, stream_copy + i, 1);

        status =
            iree_net_frame_accumulator_push_lease(accumulator, &mock.lease, 1);
        fuzz_verify_release(&mock);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }
      }

      free(stream_copy);
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 4: Empty push (boundary condition)
  //===--------------------------------------------------------------------===//

  {
    fuzz_mock_lease_t mock;
    uint8_t dummy = 0;
    fuzz_mock_lease_init(&mock, &dummy, 1);
    status = iree_net_frame_accumulator_push_lease(accumulator, &mock.lease, 0);
    iree_status_ignore(status);
    fuzz_verify_release(&mock);
  }

  //===--------------------------------------------------------------------===//
  // Test 5: Reset mid-stream (partial frame handling)
  //===--------------------------------------------------------------------===//

  if (stream_size >= 4) {
    iree_net_frame_accumulator_reset(accumulator);

    // Push partial data.
    uint8_t* partial = (uint8_t*)malloc(4);
    if (partial) {
      memcpy(partial, stream_data, 4);

      fuzz_mock_lease_t mock;
      fuzz_mock_lease_init(&mock, partial, 4);

      status =
          iree_net_frame_accumulator_push_lease(accumulator, &mock.lease, 4);
      iree_status_ignore(status);
      fuzz_verify_release(&mock);

      // Reset with partial frame in buffer.
      iree_net_frame_accumulator_reset(accumulator);

      // Verify reset worked.
      if (iree_net_frame_accumulator_has_partial_frame(accumulator)) {
        // Should not have partial frame after reset.
        abort();
      }

      free(partial);
    }
  }

  iree_net_frame_accumulator_deinitialize(accumulator);
  free(storage);

  return 0;
}
