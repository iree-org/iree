// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#version 450

// Polyfill for buffer fills that are not aligned to 4 byte offsets or lengths.
// This only implements the unaligned edges of fill operations. vkCmdFillBuffer
// should be used for the aligned interior (if any).
//
// Repeats the 4 byte value |fill_pattern| into |output_elements|, between
// |fill_offset_bytes| and |fill_offset_bytes| + |fill_length_bytes|.

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 3, binding = 0) buffer OutputBuffer { uint output_elements[]; };

layout(push_constant) uniform Constants {
  // TODO(scotttodd): low and high for 8 byte pattern
  uint fill_pattern;
  uint fill_pattern_width;  // should be 1 or 2 (or 8 later on)
  uint fill_offset_bytes;   // must be aligned to pattern width
  uint fill_length_bytes;
} input_constants;

void FillBufferUnalignedHelper(uint fill_offset_bytes, uint fill_length_bytes) {
  uint fill_aligned_offset = fill_offset_bytes % 4;
  uint fill_aligned_start_bytes = fill_offset_bytes - fill_aligned_offset;
  uint fill_aligned_start_index = fill_aligned_start_bytes / 4;

  uint shifted_pattern = 0x00000000;
  if (input_constants.fill_pattern_width == 1) {
    // Shift the pattern into each segment that is within the fill range.
    uint fill_start = fill_aligned_offset;
    uint fill_end = min(4, fill_start + fill_length_bytes);
    for (uint i = fill_start; i < fill_end; ++i) {
      shifted_pattern |= input_constants.fill_pattern << (8 * i);
    }
  } else if (input_constants.fill_pattern_width == 2) {
    // Shift the pattern into the only supported segment in the fill range.
    shifted_pattern = input_constants.fill_pattern << (8 * fill_aligned_offset);
  }
  output_elements[fill_aligned_start_index] = shifted_pattern;
}

void main() {
  uint start_byte = input_constants.fill_offset_bytes;
  uint end_byte =
      input_constants.fill_offset_bytes + input_constants.fill_length_bytes;

  // Unaligned start fill, if needed.
  if (start_byte % 4 != 0 || input_constants.fill_length_bytes < 4) {
    FillBufferUnalignedHelper(start_byte, input_constants.fill_length_bytes);
  }
  // Unaligned end fill, if needed.
  if ((end_byte % 4 != 0) &&
      (start_byte % 4 + input_constants.fill_length_bytes > 4)) {
    uint end_rounded_down = (end_byte / 4) * 4;
    uint length_end = end_byte - end_rounded_down;
    FillBufferUnalignedHelper(end_rounded_down, length_end);
  }
}
