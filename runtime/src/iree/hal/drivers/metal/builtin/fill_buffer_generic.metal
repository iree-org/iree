// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Polyfill Metal kernels for buffer fills without aligned offsets / 4-byte patterns.

struct FillSpec {
  uint64_t buffer_offset;  // Buffer offset to fill (in bytes)
  uint64_t buffer_length;  // Buffer length to fill (in bytes)
  uint32_t pattern;        // 32-bit fill pattern
};

// Fills target |buffer| with the given |spec|ification.
//
// The target |buffer| is assumed to have 16-byte aligned offset/length.
// Each thread fills one 4-compoment 32-bit element vector.
kernel void fill_buffer_16byte(
  device uint4 *buffer [[buffer(0)]],
  constant FillSpec &spec [[buffer(1)]],
  uint id [[thread_position_in_grid]]
) {
  uint64_t end = spec.buffer_length / 16;
  if (id >= end) return;
  uint64_t start = spec.buffer_offset / 16;
  buffer[start + id] = uint4(spec.pattern, spec.pattern, spec.pattern, spec.pattern);
}

// Fills target |buffer| with the given |spec|ification.
//
// The target |buffer| is assumed to have 4-byte aligned offset/length.
// Each thread fills one 32-bit scalar.
kernel void fill_buffer_4byte(
  device uint32_t *buffer [[buffer(0)]],
  constant FillSpec &spec [[buffer(1)]],
  uint id [[thread_position_in_grid]]
) {
  uint64_t end = spec.buffer_length / 4;
  if (id >= end) return;
  uint64_t start = spec.buffer_offset / 4;
  buffer[start + id] = spec.pattern;
}

// Fills target |buffer| with the given |spec|ification.
//
// The target |buffer| is assumed to have 1-byte aligned offset/length.
// Each thread fills one 8-bit scalar.
kernel void fill_buffer_1byte(
  device uint32_t *buffer [[buffer(0)]],
  constant FillSpec &spec [[buffer(1)]],
  uint id [[thread_position_in_grid]]
) {
  // We split the full buffer fill range into three parts:
  // 1. Left bytes: containing (0 to 3) bytes before the first 4-byte aligned address
  // 2. Middle bytes: aligned 32-bit scalars in the middle
  // 3. Right bytes: containing (0 to 3) bytes since the last 4-byte aligned address
  //
  // Threads are distributed from the perspecitve of handling middle 32-bit scalars.
  // We use the first thread to *additionally* handle left and right bytes.
  uint8_t left_byte_count = spec.buffer_offset % 4;
  uint8_t right_byte_count = (spec.buffer_offset + spec.buffer_length) % 4;
  uint32_t middle_pattern = metal::rotate(spec.pattern, uint32_t(8 * left_byte_count));

  // Masks for left bytes and right bytes from rotated pattern. Note that for *little
  // endian*, left bytes will replace high bits of the leftmost touched 32-bit scalar,
  // while right bytes will replace low bits of the rightmost touched 32-bit scalar.
  uint32_t left_mask = ~((uint64_t(1) << (8 * (4 - left_byte_count))) - 1);
  uint32_t right_mask = (uint64_t(1) << (8 * right_byte_count)) - 1;

  // Indexing start points in |buffer| for the threee parts.
  uint64_t left_start = spec.buffer_offset / 4;
  uint64_t middle_start = (spec.buffer_offset + 3) / 4;
  uint64_t right_start = (spec.buffer_offset + spec.buffer_length) / 4;

  if (middle_start < right_start) {  // Middle bytes
    if (middle_start + id >= right_start) return;
    buffer[middle_start + id] = middle_pattern;
  }

  if (left_byte_count != 0 && id == 0) {  // Left bytes
    uint32_t old = buffer[left_start];
    buffer[left_start] = (old & (~left_mask)) | (middle_pattern & left_mask);
  }

  if (right_byte_count != 0 && id == 0) { // Right bytes
    uint32_t old = buffer[right_start];
    buffer[right_start] = (old & (~right_mask)) | (middle_pattern & right_mask);
  }
}
