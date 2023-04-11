// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_DISPATCH_UTIL_H_
#define IREE_VM_BYTECODE_DISPATCH_UTIL_H_

#include <assert.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/bytecode/module_impl.h"
#include "iree/vm/bytecode/utils/isa.h"

//===----------------------------------------------------------------------===//
// Shared data structures
//===----------------------------------------------------------------------===//
//
// Register bounds checking
// ------------------------
// All accesses into the register lists are truncated to the valid range for the
// typed bank. This allows us to directly use the register ordinals from the
// bytecode without needing to perform any validation at load-time or run-time.
// The worst that can happen is that the bytecode program being executed doesn't
// work as intended - which, with a working compiler, shouldn't happen. Though
// there are cases where the runtime produces the register values and may know
// that they are in range it's a good habit to always mask the ordinal by the
// type-specific mask so that it's not possible for out of bounds accesses to
// sneak in. The iree_vm_registers_t struct is often kept in cache and the
// masking is cheap relative to any other validation we could be performing.
//
// Alternative register widths
// ---------------------------
// Registers in the VM are just a blob of memory and not physical device
// registers. They have a natural width of 32-bits as that covers a majority of
// our usage for i32/f32 but can be accessed at larger widths such as 64-bits or
// more for vector operations. The base of each frame's register memory is
// 16-byte aligned and accessing any individual register as a 32-bit value is
// always 4-byte aligned.
//
// Supporting other register widths is "free" in that the registers for all
// widths alias the same register storage memory. This is similar to how
// physical registers work in x86 where each register can be accessed at
// different sizes (like EAX/RAX alias and the SIMD registers alias as XMM1 is
// 128-bit, YMM1 is 256-bit, and ZMM1 is 512-bit but all the same storage).
//
// The requirements for doing this is that the base alignment for any register
// must be a multiple of 4 (due to the native 32-bit storage) AND aligned to the
// natural size of the register (so 8 bytes for i64, 16 bytes for v128, etc).
// This alignment can easily be done by masking off the low bits such that we
// know for any valid `reg` ordinal aligned to 4 bytes `reg/N` will still be
// within register storage. For example, i64 registers are accessed as `reg&~1`
// to align to 8 bytes starting at byte 0 of the register storage.
//
// Transferring between register types can be done with vm.ext.* and vm.trunc.*
// ops. For example, vm.trunc.i64.i32 will read an 8 byte register and write a
// two 4 byte registers (effectively) with hi=0 and lo=the lower 32-bits of the
// value.

// Pointers to typed register storage.
typedef struct iree_vm_registers_t {
  // 16-byte aligned i32 register array.
  int32_t* i32;
  // Naturally aligned ref register array.
  iree_vm_ref_t* ref;
} iree_vm_registers_t;

// Storage associated with each stack frame of a bytecode function.
// NOTE: we cannot store pointers to the stack in here as the stack may be
// reallocated.
typedef struct iree_vm_bytecode_frame_storage_t {
  // Calling convention results fragment.
  iree_string_view_t cconv_results;

  // Pointer to a register list within the stack frame where return registers
  // will be stored by callees upon return.
  const iree_vm_register_list_t* return_registers;

  // Counts of each register type and their relative byte offsets from the head
  // of this struct.
  uint32_t i32_register_count;
  uint32_t i32_register_offset;
  uint32_t ref_register_count;
  uint32_t ref_register_offset;
} iree_vm_bytecode_frame_storage_t;

// Maps a type ID to a type def with clamping for out of bounds values.
static inline const iree_vm_type_def_t* iree_vm_map_type(
    iree_vm_bytecode_module_t* module, int32_t type_id) {
  type_id = type_id >= module->type_count ? 0 : type_id;
  return &module->type_table[type_id];
}

//===----------------------------------------------------------------------===//
// Debugging utilities
//===----------------------------------------------------------------------===//

#if IREE_VM_EXECUTION_TRACING_FORCE_ENABLE
#define IREE_IS_DISPATCH_TRACING_ENABLED() true
#else
#define IREE_IS_DISPATCH_TRACING_ENABLED()   \
  !!(iree_vm_stack_invocation_flags(stack) & \
     IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION)
#endif  // IREE_VM_EXECUTION_TRACING_FORCE_ENABLE

#if IREE_VM_EXECUTION_TRACING_ENABLE
#define IREE_DISPATCH_TRACE_INSTRUCTION(pc_offset, op_name)  \
  if (IREE_IS_DISPATCH_TRACING_ENABLED()) {                  \
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_trace_disassembly( \
        current_frame, (pc - (pc_offset)), &regs, stderr));  \
  }

#else
#define IREE_DISPATCH_TRACE_INSTRUCTION(...)
#endif  // IREE_VM_EXECUTION_TRACING_ENABLE

#if defined(IREE_COMPILER_CLANG) && \
    IREE_VM_BYTECODE_DISPATCH_COMPUTED_GOTO_ENABLE
#define IREE_DISPATCH_MODE_COMPUTED_GOTO 1
#else
#define IREE_DISPATCH_MODE_SWITCH 1
#endif  // IREE_VM_BYTECODE_DISPATCH_COMPUTED_GOTO_ENABLE

//===----------------------------------------------------------------------===//
// Utilities matching the tablegen op encoding scheme
//===----------------------------------------------------------------------===//
// These utilities match the VM_Enc* statements in VMBase.td 1:1, allowing us
// to have the inverse of the encoding which make things easier to read.
//
// Each macro will increment the pc by the number of bytes read and as such must
// be called in the same order the values are encoded.

#define VM_DecConstI8(name) \
  OP_I8(0);                 \
  ++pc;
#define VM_DecConstI32(name) \
  OP_I32(0);                 \
  pc += 4;
#define VM_DecConstI64(name) \
  OP_I64(0);                 \
  pc += 8;
#define VM_DecConstF32(name) \
  OP_F32(0);                 \
  pc += 4;
#define VM_DecConstF64(name) \
  OP_F64(0);                 \
  pc += 8;
#define VM_DecFuncAttr(name) VM_DecConstI32(name)
#define VM_DecGlobalAttr(name) VM_DecConstI32(name)
#define VM_DecRodataAttr(name) VM_DecConstI32(name)
#define VM_DecType(name)               \
  iree_vm_map_type(module, OP_I32(0)); \
  pc += 4;
#define VM_DecTypeOf(name) VM_DecType(name)
#define VM_DecIntAttr32(name) VM_DecConstI32(name)
#define VM_DecIntAttr64(name) VM_DecConstI64(name)
#define VM_DecFloatAttr32(name) VM_DecConstF32(name)
#define VM_DecFloatAttr64(name) VM_DecConstF64(name)
#define VM_DecStrAttr(name, out_str)                     \
  (out_str)->size = (iree_host_size_t)OP_I16(0);         \
  (out_str)->data = (const char*)&bytecode_data[pc + 2]; \
  pc += 2 + (out_str)->size;
#define VM_DecBranchTarget(block_name) VM_DecConstI32(name)
#define VM_DecBranchOperands(operands_name) \
  VM_DecBranchOperandsImpl(bytecode_data, &pc)
static inline const iree_vm_register_remap_list_t* VM_DecBranchOperandsImpl(
    const uint8_t* IREE_RESTRICT bytecode_data, iree_vm_source_offset_t* pc) {
  VM_AlignPC(*pc, IREE_REGISTER_ORDINAL_SIZE);
  const iree_vm_register_remap_list_t* list =
      (const iree_vm_register_remap_list_t*)&bytecode_data[*pc];
  *pc = *pc + IREE_REGISTER_ORDINAL_SIZE +
        list->size * 2 * IREE_REGISTER_ORDINAL_SIZE;
  return list;
}
#define VM_DecOperandRegI32(name) \
  regs_i32[OP_I16(0)];            \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecOperandRegI64(name)    \
  *((int64_t*)&regs_i32[OP_I16(0)]); \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecOperandRegI64HostSize(name) \
  (iree_host_size_t) VM_DecOperandRegI64(name)
#define VM_DecOperandRegF32(name)  \
  *((float*)&regs_i32[OP_I16(0)]); \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecOperandRegF64(name)   \
  *((double*)&regs_i32[OP_I16(0)]); \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecOperandRegRef(name, out_is_move)                      \
  &regs_ref[OP_I16(0) & IREE_REF_REGISTER_MASK];                    \
  *(out_is_move) = 0; /*= OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT;*/ \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecVariadicOperands(name) \
  VM_DecVariadicOperandsImpl(bytecode_data, &pc)
static inline const iree_vm_register_list_t* VM_DecVariadicOperandsImpl(
    const uint8_t* IREE_RESTRICT bytecode_data, iree_vm_source_offset_t* pc) {
  VM_AlignPC(*pc, IREE_REGISTER_ORDINAL_SIZE);
  const iree_vm_register_list_t* list =
      (const iree_vm_register_list_t*)&bytecode_data[*pc];
  *pc = *pc + IREE_REGISTER_ORDINAL_SIZE +
        list->size * IREE_REGISTER_ORDINAL_SIZE;
  return list;
}
#define VM_DecResultRegI32(name) \
  &regs_i32[OP_I16(0)];          \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecResultRegI64(name)    \
  ((int64_t*)&regs_i32[OP_I16(0)]); \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecResultRegF32(name)  \
  ((float*)&regs_i32[OP_I16(0)]); \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecResultRegF64(name)   \
  ((double*)&regs_i32[OP_I16(0)]); \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecResultRegRef(name, out_is_move)                       \
  &regs_ref[OP_I16(0) & IREE_REF_REGISTER_MASK];                    \
  *(out_is_move) = 0; /*= OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT;*/ \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_DecVariadicResults(name) VM_DecVariadicOperands(name)

#define IREE_VM_BLOCK_MARKER_SIZE 1

//===----------------------------------------------------------------------===//
// Dispatch table structure
//===----------------------------------------------------------------------===//
// We support both computed goto (gcc/clang) and switch-based dispatch. Computed
// goto is preferred when available as it has the most efficient codegen. MSVC
// doesn't support it, though, and there may be other targets (like wasm) that
// can only handle the switch-based approach.

#if defined(IREE_DISPATCH_MODE_COMPUTED_GOTO)

// Dispatch table mapping 1:1 with bytecode ops.
// Each entry is a label within this function that can be used for computed
// goto. You can find more information on computed goto here:
// https://eli.thegreenplace.net/2012/07/12/computed-goto-for-efficient-dispatch-tables
//
// Note that we ensure the table is 256 elements long exactly to make sure
// that unused opcodes are handled gracefully.
//
// Computed gotos are pretty much the best way to dispatch interpreters but are
// not part of the C standard; GCC and clang support them but MSVC does not.
// Because the performance difference is significant we support both here but
// prefer the computed goto path where available. Empirical data shows them to
// still be a win in 2019 on x64 desktops and arm32/arm64 mobile devices.
#define BEGIN_DISPATCH_CORE()                     \
  goto* kDispatchTable_CORE[bytecode_data[pc++]]; \
  while (1)
#define END_DISPATCH_CORE()

#define DECLARE_DISPATCH_CORE_OPC(ordinal, name) &&_dispatch_CORE_##name,
#define DECLARE_DISPATCH_CORE_RSV(ordinal) &&_dispatch_unhandled,
#define DEFINE_DISPATCH_TABLE_CORE()                                    \
  static const void* kDispatchTable_CORE[256] = {IREE_VM_OP_CORE_TABLE( \
      DECLARE_DISPATCH_CORE_OPC, DECLARE_DISPATCH_CORE_RSV)};

#define DECLARE_DISPATCH_EXT_RSV(ordinal) &&_dispatch_unhandled,
#if IREE_VM_EXT_F32_ENABLE
#define DECLARE_DISPATCH_EXT_F32_OPC(ordinal, name) &&_dispatch_EXT_F32_##name,
#define DEFINE_DISPATCH_TABLE_EXT_F32()                                       \
  static const void* kDispatchTable_EXT_F32[256] = {IREE_VM_OP_EXT_F32_TABLE( \
      DECLARE_DISPATCH_EXT_F32_OPC, DECLARE_DISPATCH_EXT_RSV)};
#else
#define DEFINE_DISPATCH_TABLE_EXT_F32()
#endif  // IREE_VM_EXT_F32_ENABLE
#if IREE_VM_EXT_F64_ENABLE
#define DECLARE_DISPATCH_EXT_F64_OPC(ordinal, name) &&_dispatch_EXT_F64_##name,
#define DEFINE_DISPATCH_TABLE_EXT_F64()                                       \
  static const void* kDispatchTable_EXT_F64[256] = {IREE_VM_OP_EXT_F64_TABLE( \
      DECLARE_DISPATCH_EXT_F64_OPC, DECLARE_DISPATCH_EXT_RSV)};
#else
#define DEFINE_DISPATCH_TABLE_EXT_F64()
#endif  // IREE_VM_EXT_F64_ENABLE

#define DEFINE_DISPATCH_TABLES()   \
  DEFINE_DISPATCH_TABLE_CORE();    \
  DEFINE_DISPATCH_TABLE_EXT_F32(); \
  DEFINE_DISPATCH_TABLE_EXT_F64();

#define DISPATCH_UNHANDLED_CORE()                                           \
  _dispatch_unhandled : {                                                   \
    IREE_ASSERT(0);                                                         \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unhandled opcode"); \
  }
#define UNHANDLED_DISPATCH_PREFIX(op_name, ext)                    \
  _dispatch_CORE_##op_name : {                                     \
    IREE_ASSERT(0);                                                \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,             \
                            "unhandled dispatch extension " #ext); \
  }

#define DISPATCH_OP(ext, op_name, body)                               \
  _dispatch_##ext##_##op_name:;                                       \
  IREE_DISPATCH_TRACE_INSTRUCTION(IREE_VM_PC_OFFSET_##ext, #op_name); \
  body;                                                               \
  goto* kDispatchTable_CORE[bytecode_data[pc++]];

#define BEGIN_DISPATCH_PREFIX(op_name, ext)                                   \
  _dispatch_CORE_##op_name : goto* kDispatchTable_##ext[bytecode_data[pc++]]; \
  while (1)
#define END_DISPATCH_PREFIX() goto* kDispatchTable_CORE[bytecode_data[pc++]];

#else

// Switch-based dispatch. This is strictly less efficient than the computed
// goto approach above but is universally supported.

#define BEGIN_DISPATCH_CORE() \
  while (1) {                 \
    switch (bytecode_data[pc++])
#define END_DISPATCH_CORE() }

#define DEFINE_DISPATCH_TABLES()

#define DISPATCH_UNHANDLED_CORE()                         \
  default: {                                              \
    IREE_ASSERT(0);                                       \
    IREE_BUILTIN_UNREACHABLE(); /* ok because verified */ \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,    \
                            "unhandled core opcode");     \
  }
#define UNHANDLED_DISPATCH_PREFIX(op_name, ext)                    \
  case IREE_VM_OP_CORE_##op_name: {                                \
    IREE_ASSERT(0);                                                \
    IREE_BUILTIN_UNREACHABLE(); /* ok because verified */          \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,             \
                            "unhandled dispatch extension " #ext); \
  }

#define DISPATCH_OP(ext, op_name, body)                                 \
  case IREE_VM_OP_##ext##_##op_name: {                                  \
    IREE_DISPATCH_TRACE_INSTRUCTION(IREE_VM_PC_OFFSET_##ext, #op_name); \
    body;                                                               \
  } break;

#define BEGIN_DISPATCH_PREFIX(op_name, ext) \
  case IREE_VM_OP_CORE_##op_name: {         \
    switch (bytecode_data[pc++])
#define END_DISPATCH_PREFIX() \
  break;                      \
  }

#endif  // IREE_DISPATCH_MODE_COMPUTED_GOTO

// Common dispatch op macros

#define DISPATCH_OP_CORE_UNARY_I32(op_name, op_func)  \
  DISPATCH_OP(CORE, op_name, {                        \
    int32_t operand = VM_DecOperandRegI32("operand"); \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = op_func(operand);                       \
  });

#define DISPATCH_OP_CORE_UNARY_I64(op_name, op_func)  \
  DISPATCH_OP(CORE, op_name, {                        \
    int64_t operand = VM_DecOperandRegI64("operand"); \
    int64_t* result = VM_DecResultRegI64("result");   \
    *result = op_func(operand);                       \
  });

#define DISPATCH_OP_CORE_BINARY_I32(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                        \
    int32_t lhs = VM_DecOperandRegI32("lhs");         \
    int32_t rhs = VM_DecOperandRegI32("rhs");         \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = op_func(lhs, rhs);                      \
  });

#define DISPATCH_OP_CORE_BINARY_I64(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                        \
    int64_t lhs = VM_DecOperandRegI64("lhs");         \
    int64_t rhs = VM_DecOperandRegI64("rhs");         \
    int64_t* result = VM_DecResultRegI64("result");   \
    *result = op_func(lhs, rhs);                      \
  });

#define DISPATCH_OP_CORE_TERNARY_I32(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                         \
    int32_t a = VM_DecOperandRegI32("a");              \
    int32_t b = VM_DecOperandRegI32("b");              \
    int32_t c = VM_DecOperandRegI32("c");              \
    int32_t* result = VM_DecResultRegI32("result");    \
    *result = op_func(a, b, c);                        \
  });

#define DISPATCH_OP_CORE_TERNARY_I64(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                         \
    int64_t a = VM_DecOperandRegI64("a");              \
    int64_t b = VM_DecOperandRegI64("b");              \
    int64_t c = VM_DecOperandRegI64("c");              \
    int64_t* result = VM_DecResultRegI64("result");    \
    *result = op_func(a, b, c);                        \
  });

#define DISPATCH_OP_EXT_F32_UNARY_F32(op_name, op_func) \
  DISPATCH_OP(EXT_F32, op_name, {                       \
    float operand = VM_DecOperandRegF32("operand");     \
    float* result = VM_DecResultRegF32("result");       \
    *result = op_func(operand);                         \
  });

#define DISPATCH_OP_EXT_F32_BINARY_F32(op_name, op_func) \
  DISPATCH_OP(EXT_F32, op_name, {                        \
    float lhs = VM_DecOperandRegF32("lhs");              \
    float rhs = VM_DecOperandRegF32("rhs");              \
    float* result = VM_DecResultRegF32("result");        \
    *result = op_func(lhs, rhs);                         \
  });

#define DISPATCH_OP_EXT_F32_TERNARY_F32(op_name, op_func) \
  DISPATCH_OP(EXT_F32, op_name, {                         \
    float a = VM_DecOperandRegF32("a");                   \
    float b = VM_DecOperandRegF32("b");                   \
    float c = VM_DecOperandRegF32("c");                   \
    float* result = VM_DecResultRegF32("result");         \
    *result = op_func(a, b, c);                           \
  });

#define DISPATCH_OP_EXT_F64_UNARY_F64(op_name, op_func) \
  DISPATCH_OP(EXT_F64, op_name, {                       \
    double operand = VM_DecOperandRegF64("operand");    \
    double* result = VM_DecResultRegF64("result");      \
    *result = op_func(operand);                         \
  });

#define DISPATCH_OP_EXT_F64_BINARY_F64(op_name, op_func) \
  DISPATCH_OP(EXT_F64, op_name, {                        \
    double lhs = VM_DecOperandRegF64("lhs");             \
    double rhs = VM_DecOperandRegF64("rhs");             \
    double* result = VM_DecResultRegF64("result");       \
    *result = op_func(lhs, rhs);                         \
  });

#define DISPATCH_OP_EXT_F64_TERNARY_F64(op_name, op_func) \
  DISPATCH_OP(EXT_F64, op_name, {                         \
    double a = VM_DecOperandRegF64("a");                  \
    double b = VM_DecOperandRegF64("b");                  \
    double c = VM_DecOperandRegF64("c");                  \
    double* result = VM_DecResultRegF64("result");        \
    *result = op_func(a, b, c);                           \
  });

#endif  // IREE_VM_BYTECODE_DISPATCH_UTIL_H_
