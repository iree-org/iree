// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block command buffer instruction set for the WebGPU HAL driver.
//
// HAL commands compile into a compact uint32 instruction stream that a JS-side
// processor walks in a single bridge call. This eliminates per-command wasm↔JS
// overhead (one bridge call per stream rather than per command) and enables
// memoized command buffers where the instruction stream is shipped to JS once
// and only the binding table is refreshed per issue.
//
// Instructions operate on two WebGPU API surfaces:
//   Queue surface:   queue.writeBuffer (UPDATE_BUFFER)
//   Encoder surface: command encoder operations (COPY, FILL, DISPATCH)
//
// The builder auto-inserts ENCODER_BEGIN/END instructions to batch adjacent
// encoder commands into shared GPUCommandEncoder sessions and flush them before
// queue surface transitions.
//
// All buffer references in the instruction stream use slot indices into a
// binding table. The binding table is split into dynamic slots [0,
// dynamic_count) resolved per issue, and static slots [dynamic_count,
// total_count) resolved once at recording creation. This allows reusable
// command buffers to pay zero cost for static bindings in steady state.
//
// All buffer offsets and lengths are uint32. No shipping WebGPU implementation
// supports buffers larger than 4GB (spec minimum maxBufferSize is 256MB,
// practical limit is 2-4GB).

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_ISA_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_ISA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Instruction header encoding
//===----------------------------------------------------------------------===//

// Every instruction begins with a single uint32 header word:
//   bits [7:0]   opcode
//   bits [15:8]  flags (opcode-specific)
//   bits [31:16] size_words (total instruction size including header)
//
// Maximum instruction size: 65535 words = 256KB. The largest instruction is
// UPDATE_BUFFER with 64KB inline data = ~16K words, well within range.

#define IREE_HAL_WEBGPU_ISA_HEADER_OPCODE_MASK 0x000000FFu
#define IREE_HAL_WEBGPU_ISA_HEADER_FLAGS_MASK 0x0000FF00u
#define IREE_HAL_WEBGPU_ISA_HEADER_FLAGS_SHIFT 8
#define IREE_HAL_WEBGPU_ISA_HEADER_SIZE_MASK 0xFFFF0000u
#define IREE_HAL_WEBGPU_ISA_HEADER_SIZE_SHIFT 16

static inline uint32_t iree_hal_webgpu_isa_header_encode(uint32_t opcode,
                                                         uint32_t flags,
                                                         uint32_t size_words) {
  return (opcode & IREE_HAL_WEBGPU_ISA_HEADER_OPCODE_MASK) |
         ((flags << IREE_HAL_WEBGPU_ISA_HEADER_FLAGS_SHIFT) &
          IREE_HAL_WEBGPU_ISA_HEADER_FLAGS_MASK) |
         ((size_words << IREE_HAL_WEBGPU_ISA_HEADER_SIZE_SHIFT) &
          IREE_HAL_WEBGPU_ISA_HEADER_SIZE_MASK);
}

static inline uint32_t iree_hal_webgpu_isa_header_opcode(uint32_t header) {
  return header & IREE_HAL_WEBGPU_ISA_HEADER_OPCODE_MASK;
}

static inline uint32_t iree_hal_webgpu_isa_header_flags(uint32_t header) {
  return (header & IREE_HAL_WEBGPU_ISA_HEADER_FLAGS_MASK) >>
         IREE_HAL_WEBGPU_ISA_HEADER_FLAGS_SHIFT;
}

static inline uint32_t iree_hal_webgpu_isa_header_size_words(uint32_t header) {
  return (header & IREE_HAL_WEBGPU_ISA_HEADER_SIZE_MASK) >>
         IREE_HAL_WEBGPU_ISA_HEADER_SIZE_SHIFT;
}

//===----------------------------------------------------------------------===//
// Opcodes
//===----------------------------------------------------------------------===//

// Queue surface: host data → GPU buffer via queue.writeBuffer().
#define IREE_HAL_WEBGPU_ISA_OP_UPDATE_BUFFER 0x01

// Control: create/finish GPUCommandEncoder.
#define IREE_HAL_WEBGPU_ISA_OP_ENCODER_BEGIN 0x10
#define IREE_HAL_WEBGPU_ISA_OP_ENCODER_END 0x11

// Encoder surface: GPU→GPU copy via encoder.copyBufferToBuffer() when aligned,
// or dispatch of copy builtin shader when unaligned.
#define IREE_HAL_WEBGPU_ISA_OP_COPY_BUFFER 0x12

// Encoder surface: fill via dispatch of fill builtin shader (WebGPU has no
// native fill command).
#define IREE_HAL_WEBGPU_ISA_OP_FILL_BUFFER 0x13

// Encoder surface: user compute dispatch.
#define IREE_HAL_WEBGPU_ISA_OP_DISPATCH 0x15

// Control: execution barrier. No-op in WebGPU (GPU handles ordering within a
// single queue). Reserved for future signal/wait extensions.
#define IREE_HAL_WEBGPU_ISA_OP_BARRIER 0x20

// Control: end of instruction stream. Flushes pending GPUCommandBuffers via
// queue.submit().
#define IREE_HAL_WEBGPU_ISA_OP_RETURN 0xFF

//===----------------------------------------------------------------------===//
// Per-opcode flags
//===----------------------------------------------------------------------===//

// UPDATE_BUFFER flag: all offsets and lengths are 4-byte aligned. When set and
// the runtime base_offset is also aligned, the processor can use
// queue.writeBuffer() directly from wasm memory. Otherwise a staging buffer
// and copy compute shader are required.
#define IREE_HAL_WEBGPU_ISA_UPDATE_FLAG_ALIGNED 0x01

//===----------------------------------------------------------------------===//
// Instruction layouts
//===----------------------------------------------------------------------===//
//
// These structs document the wire format of each instruction as a sequence of
// uint32 words following the header. They are not used directly in code (the
// builder emits raw uint32 words and the JS processor reads raw uint32 words),
// but serve as the authoritative specification of the binary encoding.
//
// In all layouts:
//   - "slot" is an index into the binding table.
//   - "offset" is a byte offset baked into the instruction at recording time.
//     The effective GPU byte offset is: binding_table[slot].base_offset +
//     offset.
//   - For static slots, base_offset is always 0 (the instruction offset
//   includes
//     the full subspan offset).
//   - For dynamic slots, base_offset comes from the HAL binding table entry at
//     issue time, and the instruction offset is the caller's offset only.

// UPDATE_BUFFER (0x01) — size_words = 4 + ceil(length / 4)
//
// Writes inline host data to a GPU buffer. The data is padded to a uint32
// boundary in the instruction stream.
//
// [0] header
// [1] dst_slot
// [2] dst_offset
// [3] length (max 64KB per HAL contract)
// [4..] inline host data (padded to uint32 boundary)
typedef struct iree_hal_webgpu_isa_update_buffer_t {
  uint32_t header;
  uint32_t dst_slot;
  uint32_t dst_offset;
  uint32_t length;
  // uint32_t data[]; — variable-length inline data follows
} iree_hal_webgpu_isa_update_buffer_t;

// ENCODER_BEGIN (0x10) — size_words = 1
//
// [0] header
//
// Auto-inserted by builder before the first encoder command in a sequence.
// Processor creates a GPUCommandEncoder.

// ENCODER_END (0x11) — size_words = 1
//
// [0] header
//
// Auto-inserted by builder when a queue command follows encoder commands, or at
// finalize. Processor calls encoder.finish() and pushes the resulting
// GPUCommandBuffer to a pending array.

// COPY_BUFFER (0x12) — size_words = 6
//
// [0] header
// [1] src_slot
// [2] src_offset
// [3] dst_slot
// [4] dst_offset
// [5] length
//
// Processor checks runtime alignment of (src_base + src_offset),
// (dst_base + dst_offset), and length. If all are 4-byte aligned, uses
// encoder.copyBufferToBuffer(). Otherwise dispatches the copy builtin shader.
typedef struct iree_hal_webgpu_isa_copy_buffer_t {
  uint32_t header;
  uint32_t src_slot;
  uint32_t src_offset;
  uint32_t dst_slot;
  uint32_t dst_offset;
  uint32_t length;
} iree_hal_webgpu_isa_copy_buffer_t;

// FILL_BUFFER (0x13) — size_words = 6
//
// [0] header
// [1] dst_slot
// [2] dst_offset
// [3] length
// [4] pattern (pre-replicated to uint32: 1B×4, 2B×2, 4B as-is)
// [5] pattern_length (original 1, 2, or 4 — for edge word handling)
//
// Always dispatches the fill builtin shader (WebGPU has no native fill).
typedef struct iree_hal_webgpu_isa_fill_buffer_t {
  uint32_t header;
  uint32_t dst_slot;
  uint32_t dst_offset;
  uint32_t length;
  uint32_t pattern;
  uint32_t pattern_length;
} iree_hal_webgpu_isa_fill_buffer_t;

// DISPATCH (0x15) — size_words = 6 + binding_count * 3
//
// [0] header
// [1] pipeline_handle
// [2] bind_group_layout_handle
// [3] workgroup_x
// [4] workgroup_y
// [5] workgroup_z
// For each binding [i]:
//   [6 + i*3 + 0] slot
//   [6 + i*3 + 1] offset
//   [6 + i*3 + 2] length
//
// binding_count = (size_words - 6) / 3
//
// Pipeline and bind group layout handles are baked at recording time from the
// executable. The processor creates a bind group from the layout and resolved
// buffer bindings, then dispatches a compute pass.
typedef struct iree_hal_webgpu_isa_dispatch_t {
  uint32_t header;
  uint32_t pipeline_handle;
  uint32_t bind_group_layout_handle;
  uint32_t workgroup_x;
  uint32_t workgroup_y;
  uint32_t workgroup_z;
  // iree_hal_webgpu_isa_dispatch_binding_t bindings[]; — variable-length
} iree_hal_webgpu_isa_dispatch_t;

// Per-binding entry within a DISPATCH instruction.
typedef struct iree_hal_webgpu_isa_dispatch_binding_t {
  uint32_t slot;
  uint32_t offset;
  uint32_t length;
} iree_hal_webgpu_isa_dispatch_binding_t;

// BARRIER (0x20) — size_words = 1
//
// [0] header
//
// No-op in the current implementation. WebGPU guarantees ordering within a
// single queue, so execution barriers are implicit. Reserved for future
// signal/wait extensions.

// RETURN (0xFF) — size_words = 1
//
// [0] header
//
// Marks end of instruction stream. Processor flushes any pending
// GPUCommandBuffers via queue.submit().

//===----------------------------------------------------------------------===//
// Binding table wire format
//===----------------------------------------------------------------------===//

// Wire format for a single binding table entry passed from wasm to JS.
// The binding table is a flat array in wasm linear memory, 2 words per entry.
//
// For static entries: gpu_buffer_handle is resolved at recording creation,
//   base_offset is 0 (instruction offsets include full subspan offset).
// For dynamic entries: gpu_buffer_handle is resolved per issue from the HAL
//   binding table, base_offset is the subspan offset from the HAL entry.
typedef struct iree_hal_webgpu_isa_binding_table_entry_t {
  uint32_t gpu_buffer_handle;
  uint32_t base_offset;
} iree_hal_webgpu_isa_binding_table_entry_t;

//===----------------------------------------------------------------------===//
// Builtins descriptor
//===----------------------------------------------------------------------===//

// Wire format for passing builtin pipeline handles to the JS processor.
// 4 uint32 handles packed contiguously in wasm linear memory.
typedef struct iree_hal_webgpu_isa_builtins_descriptor_t {
  uint32_t fill_pipeline;
  uint32_t fill_bind_group_layout;
  uint32_t copy_pipeline;
  uint32_t copy_bind_group_layout;
} iree_hal_webgpu_isa_builtins_descriptor_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_ISA_H_
