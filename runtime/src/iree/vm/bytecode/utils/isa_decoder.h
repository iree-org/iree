// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_UTILS_ISA_DECODER_H_
#define IREE_VM_BYTECODE_UTILS_ISA_DECODER_H_

// Shared bytecode field decoding utilities.
//
// This header contains small, policy-free decode helpers that operate on a
// `(bytecode_data, pc)` stream. Bounds checks and semantic validation are
// performed by higher-level consumers (verifier/dispatch/disassembler/etc).
//
// For policy-substituted macros that build on these helpers, see
// `isa_decoder.inl`.

#include "iree/vm/bytecode/utils/isa.h"

//===----------------------------------------------------------------------===//
// Decoding utilities
//===----------------------------------------------------------------------===//

static inline iree_vm_source_offset_t iree_vm_isa_align_pc(
    iree_vm_source_offset_t pc, int alignment) {
  return (pc + (alignment - 1)) & ~(alignment - 1);
}

//===----------------------------------------------------------------------===//
// Primitive field decoding (unchecked)
//===----------------------------------------------------------------------===//

static inline uint8_t iree_vm_isa_decode_u8(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  return data[(*pc)++];
}

static inline uint16_t iree_vm_isa_decode_u16(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  const uint16_t v = iree_unaligned_load_le((const uint16_t*)&data[*pc]);
  *pc += 2;
  return v;
}

static inline uint32_t iree_vm_isa_decode_u32(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  const uint32_t v = iree_unaligned_load_le((const uint32_t*)&data[*pc]);
  *pc += 4;
  return v;
}

static inline uint64_t iree_vm_isa_decode_u64(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  const uint64_t v = iree_unaligned_load_le((const uint64_t*)&data[*pc]);
  *pc += 8;
  return v;
}

static inline float iree_vm_isa_decode_f32(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  const float v = iree_unaligned_load_le((const float*)&data[*pc]);
  *pc += 4;
  return v;
}

static inline double iree_vm_isa_decode_f64(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  const double v = iree_unaligned_load_le((const double*)&data[*pc]);
  *pc += 8;
  return v;
}

static inline iree_string_view_t iree_vm_isa_decode_string_view(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  const uint16_t n = iree_vm_isa_decode_u16(data, pc);
  iree_string_view_t v = iree_make_string_view((const char*)&data[*pc], n);
  *pc += n;
  return v;
}

//===----------------------------------------------------------------------===//
// Aggregate field decoding helpers (unchecked)
//===----------------------------------------------------------------------===//

// Decodes a variadic register list encoded in the bytecode stream.
// The returned pointer aliases |data| and is only valid as long as |data| is.
static inline const iree_vm_register_list_t*
iree_vm_isa_decode_register_list_unchecked(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  *pc = iree_vm_isa_align_pc(*pc, IREE_REGISTER_ORDINAL_SIZE);
  const iree_vm_register_list_t* list =
      (const iree_vm_register_list_t*)&data[*pc];
  *pc = *pc + IREE_REGISTER_ORDINAL_SIZE +
        list->size * IREE_REGISTER_ORDINAL_SIZE;
  return list;
}

// Decodes an interleaved src-dst remap list encoded in the bytecode stream.
// The returned pointer aliases |data| and is only valid as long as |data| is.
static inline const iree_vm_register_remap_list_t*
iree_vm_isa_decode_register_remap_list_unchecked(
    const uint8_t* IREE_RESTRICT data,
    iree_vm_source_offset_t* IREE_RESTRICT pc) {
  *pc = iree_vm_isa_align_pc(*pc, IREE_REGISTER_ORDINAL_SIZE);
  const iree_vm_register_remap_list_t* list =
      (const iree_vm_register_remap_list_t*)&data[*pc];
  *pc = *pc + IREE_REGISTER_ORDINAL_SIZE +
        list->size * 2 * IREE_REGISTER_ORDINAL_SIZE;
  return list;
}

#endif  // IREE_VM_BYTECODE_UTILS_ISA_DECODER_H_
