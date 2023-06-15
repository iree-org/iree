// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Polyfill Metal kernels for buffer copies without 4-byte aligned offsets / lengths.

struct CopySpec {
  uint64_t src_buffer_offset;  // Source buffer offset (in bytes)
  uint64_t dst_buffer_offset;  // Destination buffer offset (in bytes)
  uint64_t length;             // Buffer length to fill (in bytes)
};

// Copies data from |src_buffer| to |dst_buffer| with the given |spec|ification.
//
// No alignment requirement on source/destination buffer offset and length.
// Each thread copies over one byte. This won't be good for perf; so just a fallback.
kernel void copy_buffer_1byte(
  device uint8_t *src_buffer [[buffer(0)]],
  device uint8_t *dst_buffer [[buffer(1)]],
  constant CopySpec &spec [[buffer(2)]],
  uint id [[thread_position_in_grid]]
) {
  if (id >= spec.length) return;
  dst_buffer[spec.dst_buffer_offset + id] = src_buffer[spec.src_buffer_offset + id];
}
