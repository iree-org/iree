// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Polyfill Metal kernels for buffer fills without aligned offsets / 4-byte
// patterns.

struct FillSpec {
  uint64_t buffer_offset;  // Buffer offset to fill (in bytes)
  uint64_t buffer_length;  // Buffer length to fill (in bytes)
  uint32_t pattern;        // 32-bit fill pattern
};

// Fills target |buffer| with the given |spec|ification.
//
// The target |buffer| is assumed to have 16-byte aligned offset/length.
// Each thread fills one 4-component 32-bit element vector.
kernel void fill_buffer_16byte(device uint4* buffer [[buffer(0)]],
                               constant FillSpec& spec [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
  uint64_t end = spec.buffer_length / 16;
  if (id >= end) return;
  uint64_t start = spec.buffer_offset / 16;
  buffer[start + id] =
      uint4(spec.pattern, spec.pattern, spec.pattern, spec.pattern);
}

// Fills target |buffer| with the given |spec|ification.
//
// The target |buffer| is assumed to have 4-byte aligned offset/length.
// Each thread fills one 32-bit scalar.
kernel void fill_buffer_4byte(device uint32_t* buffer [[buffer(0)]],
                              constant FillSpec& spec [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
  uint64_t end = spec.buffer_length / 4;
  if (id >= end) return;
  uint64_t start = spec.buffer_offset / 4;
  buffer[start + id] = spec.pattern;
}

// Fills target |buffer| with the given |spec|ification.
//
// The target |buffer| is assumed to have 1-byte aligned offset/length.
// Each thread fills one 8-bit scalar.
kernel void fill_buffer_1byte(device uint8_t* buffer [[buffer(0)]],
                              constant FillSpec& spec [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
  if (id >= spec.buffer_length) return;
  uint8_t pattern_byte = static_cast<uint8_t>(spec.pattern >> (8 * (id & 3u)));
  buffer[spec.buffer_offset + id] = pattern_byte;
}
