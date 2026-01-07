// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Policy-substituted ISA decoding macros.
//
// This file intentionally has no include guard so that it can be included with
// different policy configurations (though most users include it once per TU).
//
// Required configuration (define before including):
// - IREE_VM_ISA_BYTECODE_DATA: expression yielding `const uint8_t*`.
// - IREE_VM_ISA_PC: lvalue of type `iree_vm_source_offset_t`.
//
// Optional policy hooks (define before including; defaults are no-ops):
// - IREE_VM_ISA_REQUIRE(bytes): statement that ensures `pc + bytes` is in
//   range (may `return` on failure).
// - IREE_VM_ISA_VALIDATE_*: statements that validate decoded fields (may
//   `return` on failure).
// - IREE_VM_ISA_TYPE_T: type used for decoded type defs (defaults to
//   `iree_vm_type_def_t`).
// - IREE_VM_ISA_LOOKUP_TYPE(type_id, out_type): statement mapping a type ID to
//   an `IREE_VM_ISA_TYPE_T` lvalue (required for `IREE_VM_ISA_DECODE_TYPE*`).
//
// Example (dispatch-style, unchecked):
//   #define IREE_VM_ISA_BYTECODE_DATA bytecode_data
//   #define IREE_VM_ISA_PC pc
//   #define IREE_VM_ISA_REQUIRE(bytes) ((void)0)
//   #define IREE_VM_ISA_LOOKUP_TYPE(type_id, out_type) \
//     do { (out_type) = iree_vm_map_type(module, (int32_t)(type_id)); } while
//     (0)
//   #include "iree/vm/bytecode/utils/isa_decoder.inl"

#include "iree/vm/bytecode/utils/isa_decoder.h"

#if !defined(IREE_VM_ISA_BYTECODE_DATA)
#error \
    "IREE_VM_ISA_BYTECODE_DATA must be defined before including isa_decoder.inl"
#endif  // !IREE_VM_ISA_BYTECODE_DATA

#if !defined(IREE_VM_ISA_PC)
#error "IREE_VM_ISA_PC must be defined before including isa_decoder.inl"
#endif  // !IREE_VM_ISA_PC

//===----------------------------------------------------------------------===//
// Policy defaults
//===----------------------------------------------------------------------===//

#if !defined(IREE_VM_ISA_REQUIRE)
#define IREE_VM_ISA_REQUIRE(bytes) ((void)0)
#endif  // !IREE_VM_ISA_REQUIRE

#if !defined(IREE_VM_ISA_TYPE_T)
#define IREE_VM_ISA_TYPE_T iree_vm_type_def_t
#endif  // !IREE_VM_ISA_TYPE_T

#if !defined(IREE_VM_ISA_VALIDATE_TYPE_ID)
#define IREE_VM_ISA_VALIDATE_TYPE_ID(type_id) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_TYPE_ID

#if !defined(IREE_VM_ISA_VALIDATE_REG_I32)
#define IREE_VM_ISA_VALIDATE_REG_I32(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_I32

#if !defined(IREE_VM_ISA_VALIDATE_REG_I64)
#define IREE_VM_ISA_VALIDATE_REG_I64(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_I64

#if !defined(IREE_VM_ISA_VALIDATE_REG_F32)
#define IREE_VM_ISA_VALIDATE_REG_F32(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_F32

#if !defined(IREE_VM_ISA_VALIDATE_REG_F64)
#define IREE_VM_ISA_VALIDATE_REG_F64(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_F64

#if !defined(IREE_VM_ISA_VALIDATE_REG_REF_ALLOW_MOVE)
#define IREE_VM_ISA_VALIDATE_REG_REF_ALLOW_MOVE(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_REF_ALLOW_MOVE

#if !defined(IREE_VM_ISA_VALIDATE_REG_REF_NO_MOVE)
#define IREE_VM_ISA_VALIDATE_REG_REF_NO_MOVE(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_REF_NO_MOVE

#if !defined(IREE_VM_ISA_VALIDATE_REG_ANY)
#define IREE_VM_ISA_VALIDATE_REG_ANY(ordinal) ((void)0)
#endif  // !IREE_VM_ISA_VALIDATE_REG_ANY

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_IMPL_READ_U8(out_value)                                \
  do {                                                                     \
    IREE_VM_ISA_REQUIRE(1);                                                \
    (out_value) =                                                          \
        iree_vm_isa_decode_u8(IREE_VM_ISA_BYTECODE_DATA, &IREE_VM_ISA_PC); \
  } while (0)

#define IREE_VM_ISA_IMPL_READ_U16(out_value)                                \
  do {                                                                      \
    IREE_VM_ISA_REQUIRE(2);                                                 \
    (out_value) =                                                           \
        iree_vm_isa_decode_u16(IREE_VM_ISA_BYTECODE_DATA, &IREE_VM_ISA_PC); \
  } while (0)

#define IREE_VM_ISA_IMPL_READ_U32(out_value)                                \
  do {                                                                      \
    IREE_VM_ISA_REQUIRE(4);                                                 \
    (out_value) =                                                           \
        iree_vm_isa_decode_u32(IREE_VM_ISA_BYTECODE_DATA, &IREE_VM_ISA_PC); \
  } while (0)

#define IREE_VM_ISA_IMPL_READ_U64(out_value)                                \
  do {                                                                      \
    IREE_VM_ISA_REQUIRE(8);                                                 \
    (out_value) =                                                           \
        iree_vm_isa_decode_u64(IREE_VM_ISA_BYTECODE_DATA, &IREE_VM_ISA_PC); \
  } while (0)

#define IREE_VM_ISA_IMPL_READ_F32(out_value)                                \
  do {                                                                      \
    IREE_VM_ISA_REQUIRE(4);                                                 \
    (out_value) =                                                           \
        iree_vm_isa_decode_f32(IREE_VM_ISA_BYTECODE_DATA, &IREE_VM_ISA_PC); \
  } while (0)

#define IREE_VM_ISA_IMPL_READ_F64(out_value)                                \
  do {                                                                      \
    IREE_VM_ISA_REQUIRE(8);                                                 \
    (out_value) =                                                           \
        iree_vm_isa_decode_f64(IREE_VM_ISA_BYTECODE_DATA, &IREE_VM_ISA_PC); \
  } while (0)

//===----------------------------------------------------------------------===//
// Core fields
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_DECODE_ALIGN_PC(alignment) \
  IREE_VM_ISA_PC = iree_vm_isa_align_pc(IREE_VM_ISA_PC, (alignment))

#define IREE_VM_ISA_DECODE_OPCODE(opcode) \
  uint8_t opcode = 0;                     \
  IREE_VM_ISA_IMPL_READ_U8(opcode)

//===----------------------------------------------------------------------===//
// Constants and attributes
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_DECODE_CONST_I8(name) \
  uint8_t name = 0;                       \
  IREE_VM_ISA_IMPL_READ_U8(name)
#define IREE_VM_ISA_DECODE_CONST_I16(name) \
  uint16_t name = 0;                       \
  IREE_VM_ISA_IMPL_READ_U16(name)
#define IREE_VM_ISA_DECODE_CONST_I32(name) \
  uint32_t name = 0;                       \
  IREE_VM_ISA_IMPL_READ_U32(name)
#define IREE_VM_ISA_DECODE_CONST_I64(name) \
  uint64_t name = 0;                       \
  IREE_VM_ISA_IMPL_READ_U64(name)
#define IREE_VM_ISA_DECODE_CONST_F32(name) \
  float name = 0.0f;                       \
  IREE_VM_ISA_IMPL_READ_F32(name)
#define IREE_VM_ISA_DECODE_CONST_F64(name) \
  double name = 0.0;                       \
  IREE_VM_ISA_IMPL_READ_F64(name)

#define IREE_VM_ISA_DECODE_ATTR_I32(name) \
  int32_t name = 0;                       \
  do {                                    \
    uint32_t __u32 = 0;                   \
    IREE_VM_ISA_IMPL_READ_U32(__u32);     \
    name = (int32_t)__u32;                \
    ((void)name);                         \
  } while (0)
#define IREE_VM_ISA_DECODE_ATTR_I64(name) \
  int64_t name = 0;                       \
  do {                                    \
    uint64_t __u64 = 0;                   \
    IREE_VM_ISA_IMPL_READ_U64(__u64);     \
    name = (int64_t)__u64;                \
    ((void)name);                         \
  } while (0)
#define IREE_VM_ISA_DECODE_ATTR_F32(name) \
  float name = 0.0f;                      \
  IREE_VM_ISA_IMPL_READ_F32(name)
#define IREE_VM_ISA_DECODE_ATTR_F64(name) \
  double name = 0.0;                      \
  IREE_VM_ISA_IMPL_READ_F64(name)

#define IREE_VM_ISA_DECODE_FUNC_ATTR(name) IREE_VM_ISA_DECODE_CONST_I32(name)
#define IREE_VM_ISA_DECODE_GLOBAL_ATTR(name) IREE_VM_ISA_DECODE_CONST_I32(name)
#define IREE_VM_ISA_DECODE_RODATA_ATTR(name) IREE_VM_ISA_DECODE_CONST_I32(name)

#define IREE_VM_ISA_DECODE_STRING_ATTR(name)                   \
  uint16_t name##_length = 0;                                  \
  IREE_VM_ISA_IMPL_READ_U16(name##_length);                    \
  IREE_VM_ISA_REQUIRE(name##_length);                          \
  iree_string_view_t name = iree_make_string_view(             \
      (const char*)&IREE_VM_ISA_BYTECODE_DATA[IREE_VM_ISA_PC], \
      (iree_host_size_t)name##_length);                        \
  ((void)name);                                                \
  IREE_VM_ISA_PC += (iree_vm_source_offset_t)name##_length

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_DECODE_TYPE_ID(name) \
  uint32_t name = 0;                     \
  IREE_VM_ISA_IMPL_READ_U32(name);       \
  IREE_VM_ISA_VALIDATE_TYPE_ID(name)

#if !defined(IREE_VM_ISA_LOOKUP_TYPE)
#define IREE_VM_ISA_LOOKUP_TYPE(type_id, out_type) \
  static_assert(                                   \
      0,                                           \
      "IREE_VM_ISA_LOOKUP_TYPE must be defined for IREE_VM_ISA_DECODE_TYPE*")
#endif  // !IREE_VM_ISA_LOOKUP_TYPE

#define IREE_VM_ISA_DECODE_TYPE(name)            \
  IREE_VM_ISA_DECODE_TYPE_ID(name##_type_id);    \
  IREE_VM_ISA_TYPE_T name;                       \
  IREE_VM_ISA_LOOKUP_TYPE(name##_type_id, name); \
  (void)(name##_type_id)

#define IREE_VM_ISA_DECODE_TYPE_OF(name) IREE_VM_ISA_DECODE_TYPE(name)

//===----------------------------------------------------------------------===//
// Register ordinals
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_DECODE_OPERAND_I32(name) \
  uint16_t name = 0;                         \
  IREE_VM_ISA_IMPL_READ_U16(name);           \
  IREE_VM_ISA_VALIDATE_REG_I32(name)

#define IREE_VM_ISA_DECODE_OPERAND_I64(name) \
  uint16_t name = 0;                         \
  IREE_VM_ISA_IMPL_READ_U16(name);           \
  IREE_VM_ISA_VALIDATE_REG_I64(name)

#define IREE_VM_ISA_DECODE_OPERAND_F32(name) \
  uint16_t name = 0;                         \
  IREE_VM_ISA_IMPL_READ_U16(name);           \
  IREE_VM_ISA_VALIDATE_REG_F32(name)

#define IREE_VM_ISA_DECODE_OPERAND_F64(name) \
  uint16_t name = 0;                         \
  IREE_VM_ISA_IMPL_READ_U16(name);           \
  IREE_VM_ISA_VALIDATE_REG_F64(name)

#define IREE_VM_ISA_DECODE_RESULT_I32(name) IREE_VM_ISA_DECODE_OPERAND_I32(name)
#define IREE_VM_ISA_DECODE_RESULT_I64(name) IREE_VM_ISA_DECODE_OPERAND_I64(name)
#define IREE_VM_ISA_DECODE_RESULT_F32(name) IREE_VM_ISA_DECODE_OPERAND_F32(name)
#define IREE_VM_ISA_DECODE_RESULT_F64(name) IREE_VM_ISA_DECODE_OPERAND_F64(name)

// Ref operand/result - rejects MOVE bit (for ops that don't support MOVE).
// Defines:
// - `<name>_ordinal`: raw encoded ordinal (includes type/move bits)
// - `<name>`: masked ordinal index for regs_ref[]
#define IREE_VM_ISA_DECODE_OPERAND_REF(name)                      \
  uint16_t name##_ordinal = 0;                                    \
  IREE_VM_ISA_IMPL_READ_U16(name##_ordinal);                      \
  IREE_VM_ISA_VALIDATE_REG_REF_NO_MOVE(name##_ordinal);           \
  const uint16_t name =                                           \
      (uint16_t)(name##_ordinal & IREE_VM_ISA_REF_REGISTER_MASK); \
  ((void)name)

#define IREE_VM_ISA_DECODE_RESULT_REF(name) IREE_VM_ISA_DECODE_OPERAND_REF(name)

// Ref operand/result - allows MOVE bit (for ops that support ownership
// transfer). Defines:
// - `<name>_ordinal`: raw encoded ordinal (includes type/move bits)
// - `<name>_is_move`: bool indicating MOVE bit
// - `<name>`: masked ordinal index for regs_ref[]
#define IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(name)                 \
  uint16_t name##_ordinal = 0;                                    \
  IREE_VM_ISA_IMPL_READ_U16(name##_ordinal);                      \
  IREE_VM_ISA_VALIDATE_REG_REF_ALLOW_MOVE(name##_ordinal);        \
  const bool name##_is_move =                                     \
      (name##_ordinal & IREE_VM_ISA_REF_REGISTER_MOVE_BIT) != 0;  \
  ((void)name##_is_move);                                         \
  const uint16_t name =                                           \
      (uint16_t)(name##_ordinal & IREE_VM_ISA_REF_REGISTER_MASK); \
  ((void)name)

#define IREE_VM_ISA_DECODE_RESULT_REF_MOVE(name) \
  IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(name)

//===----------------------------------------------------------------------===//
// Variadic aggregates
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(name)               \
  IREE_VM_ISA_DECODE_ALIGN_PC(IREE_REGISTER_ORDINAL_SIZE);       \
  IREE_VM_ISA_REQUIRE(IREE_REGISTER_ORDINAL_SIZE);               \
  const iree_vm_register_list_t* name =                          \
      (const iree_vm_register_list_t*)&IREE_VM_ISA_BYTECODE_DATA \
          [IREE_VM_ISA_PC];                                      \
  IREE_VM_ISA_PC += IREE_REGISTER_ORDINAL_SIZE;                  \
  IREE_VM_ISA_REQUIRE((name)->size* IREE_REGISTER_ORDINAL_SIZE); \
  IREE_VM_ISA_PC += (name)->size * IREE_REGISTER_ORDINAL_SIZE

#define IREE_VM_ISA_DECODE_VARIADIC_RESULTS(name) \
  IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(name)

//===----------------------------------------------------------------------===//
// Branch metadata
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(name) \
  IREE_VM_ISA_DECODE_CONST_I32(name)

#define IREE_VM_ISA_DECODE_BRANCH_OPERANDS(name)                       \
  IREE_VM_ISA_DECODE_ALIGN_PC(IREE_REGISTER_ORDINAL_SIZE);             \
  IREE_VM_ISA_REQUIRE(IREE_REGISTER_ORDINAL_SIZE);                     \
  const iree_vm_register_remap_list_t* name =                          \
      (const iree_vm_register_remap_list_t*)&IREE_VM_ISA_BYTECODE_DATA \
          [IREE_VM_ISA_PC];                                            \
  IREE_VM_ISA_PC += IREE_REGISTER_ORDINAL_SIZE;                        \
  IREE_VM_ISA_REQUIRE((name)->size * 2 * IREE_REGISTER_ORDINAL_SIZE);  \
  IREE_VM_ISA_PC += (name)->size * 2 * IREE_REGISTER_ORDINAL_SIZE

//===----------------------------------------------------------------------===//
// Operand/result mapping
//===----------------------------------------------------------------------===//

// NOTE: these macros assume locals:
//   int32_t* IREE_RESTRICT regs_i32;
//   iree_vm_ref_t* IREE_RESTRICT regs_ref;

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(name) \
  IREE_VM_ISA_DECODE_OPERAND_I32(name##_reg);         \
  const int32_t name = regs_i32[name##_reg];          \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_RESULT_I32(name)   \
  IREE_VM_ISA_DECODE_RESULT_I32(name##_reg);           \
  int32_t* IREE_RESTRICT name = &regs_i32[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(name)    \
  IREE_VM_ISA_DECODE_OPERAND_I64(name##_reg);            \
  const int64_t name = *(int64_t*)&regs_i32[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_RESULT_I64(name)             \
  IREE_VM_ISA_DECODE_RESULT_I64(name##_reg);                     \
  int64_t* IREE_RESTRICT name = (int64_t*)&regs_i32[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(name) \
  IREE_VM_ISA_DECODE_OPERAND_F32(name##_reg);         \
  const float name = *(float*)&regs_i32[name##_reg];  \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_RESULT_F32(name)         \
  IREE_VM_ISA_DECODE_RESULT_F32(name##_reg);                 \
  float* IREE_RESTRICT name = (float*)&regs_i32[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(name)  \
  IREE_VM_ISA_DECODE_OPERAND_F64(name##_reg);          \
  const double name = *(double*)&regs_i32[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_RESULT_F64(name)           \
  IREE_VM_ISA_DECODE_RESULT_F64(name##_reg);                   \
  double* IREE_RESTRICT name = (double*)&regs_i32[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64_HOST_SIZE(name) \
  IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(name##_i64);          \
  const iree_host_size_t name = (iree_host_size_t)name##_i64;   \
  ((void)name##_i64)

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_REF(name)        \
  IREE_VM_ISA_DECODE_OPERAND_REF(name##_reg);                \
  iree_vm_ref_t* IREE_RESTRICT name = &regs_ref[name##_reg]; \
  IREE_VM_ISA_DISPATCH_REF_DEBUG_CHECK(name);                \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_RESULT_REF(name)         \
  IREE_VM_ISA_DECODE_RESULT_REF(name##_reg);                 \
  iree_vm_ref_t* IREE_RESTRICT name = &regs_ref[name##_reg]; \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_OPERAND_REF_MOVE(name)   \
  IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(name##_reg);           \
  const bool name##_is_move = name##_reg##_is_move;          \
  ((void)name##_is_move);                                    \
  iree_vm_ref_t* IREE_RESTRICT name = &regs_ref[name##_reg]; \
  IREE_VM_ISA_DISPATCH_REF_DEBUG_CHECK(name);                \
  ((void)name##_reg)

#define IREE_VM_ISA_DISPATCH_DECODE_RESULT_REF_MOVE(name)    \
  IREE_VM_ISA_DECODE_RESULT_REF_MOVE(name##_reg);            \
  const bool name##_is_move = name##_reg##_is_move;          \
  ((void)name##_is_move);                                    \
  iree_vm_ref_t* IREE_RESTRICT name = &regs_ref[name##_reg]; \
  ((void)name##_reg)

// Branch target encoded as a PC (u32) converted to source offset.
#define IREE_VM_ISA_DISPATCH_DECODE_BRANCH_TARGET(name)                        \
  IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(name##_pc_u32);                          \
  const iree_vm_source_offset_t name = (iree_vm_source_offset_t)name##_pc_u32; \
  ((void)name##_pc_u32)
