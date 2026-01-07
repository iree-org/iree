// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_DISPATCH_UTIL_H_
#define IREE_VM_BYTECODE_DISPATCH_UTIL_H_

#include <assert.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
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

  // Result status code from function execution.
  // Initialized to IREE_STATUS_UNKNOWN to indicate the frame has not yet
  // completed. On successful return, set to IREE_STATUS_OK to signal that the
  // compiler-guaranteed ref cleanup has run and the frame cleanup loop can be
  // skipped. Frames that error or get unwound (due to callee errors) retain
  // the non-OK value, causing cleanup to release any remaining refs.
  iree_status_code_t result_code;

  // Counts of each register type and their relative byte offsets from the head
  // of this struct.
  uint32_t i32_register_count;
  uint32_t i32_register_offset;
  uint32_t ref_register_count;
  uint32_t ref_register_offset;
} iree_vm_bytecode_frame_storage_t;

// Maps a type ID to a type def with clamping for out of bounds values.
static inline iree_vm_type_def_t iree_vm_map_type(
    iree_vm_bytecode_module_t* module, int32_t type_id) {
  type_id = type_id >= module->type_count ? 0 : type_id;
  return module->type_table[type_id];
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
#define IREE_DISPATCH_TRACE_INSTRUCTION(pc_offset, op_name) \
  if (IREE_IS_DISPATCH_TRACING_ENABLED()) {                 \
    IREE_IGNORE_ERROR(iree_vm_bytecode_trace_disassembly(   \
        current_frame, (pc - (pc_offset)), &regs, stderr)); \
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
#define IREE_VM_ISA_DISPATCH_BEGIN_CORE()         \
  goto* kDispatchTable_CORE[bytecode_data[pc++]]; \
  while (1)
#define IREE_VM_ISA_DISPATCH_END_CORE()

#define IREE_VM_ISA_DISPATCH_DECLARE_CORE_OPC(ordinal, name) \
  &&_dispatch_CORE_##name,
#define IREE_VM_ISA_DISPATCH_DECLARE_CORE_RSV(ordinal) &&_dispatch_unhandled,
#define IREE_VM_ISA_DISPATCH_DEFINE_TABLE_CORE()                   \
  static const void* kDispatchTable_CORE[256] = {                  \
      IREE_VM_OP_CORE_TABLE(IREE_VM_ISA_DISPATCH_DECLARE_CORE_OPC, \
                            IREE_VM_ISA_DISPATCH_DECLARE_CORE_RSV)};

#define IREE_VM_ISA_DISPATCH_DECLARE_EXT_RSV(ordinal) &&_dispatch_unhandled,
#if IREE_VM_EXT_F32_ENABLE
#define IREE_VM_ISA_DISPATCH_DECLARE_EXT_F32_OPC(ordinal, name) \
  &&_dispatch_EXT_F32_##name,
#define IREE_VM_ISA_DISPATCH_DEFINE_TABLE_EXT_F32()                      \
  static const void* kDispatchTable_EXT_F32[256] = {                     \
      IREE_VM_OP_EXT_F32_TABLE(IREE_VM_ISA_DISPATCH_DECLARE_EXT_F32_OPC, \
                               IREE_VM_ISA_DISPATCH_DECLARE_EXT_RSV)};
#else
#define IREE_VM_ISA_DISPATCH_DEFINE_TABLE_EXT_F32()
#endif  // IREE_VM_EXT_F32_ENABLE
#if IREE_VM_EXT_F64_ENABLE
#define IREE_VM_ISA_DISPATCH_DECLARE_EXT_F64_OPC(ordinal, name) \
  &&_dispatch_EXT_F64_##name,
#define IREE_VM_ISA_DISPATCH_DEFINE_TABLE_EXT_F64()                      \
  static const void* kDispatchTable_EXT_F64[256] = {                     \
      IREE_VM_OP_EXT_F64_TABLE(IREE_VM_ISA_DISPATCH_DECLARE_EXT_F64_OPC, \
                               IREE_VM_ISA_DISPATCH_DECLARE_EXT_RSV)};
#else
#define IREE_VM_ISA_DISPATCH_DEFINE_TABLE_EXT_F64()
#endif  // IREE_VM_EXT_F64_ENABLE

#define IREE_VM_ISA_DISPATCH_DEFINE_TABLES()   \
  IREE_VM_ISA_DISPATCH_DEFINE_TABLE_CORE();    \
  IREE_VM_ISA_DISPATCH_DEFINE_TABLE_EXT_F32(); \
  IREE_VM_ISA_DISPATCH_DEFINE_TABLE_EXT_F64();

#define IREE_VM_ISA_DISPATCH_UNHANDLED_CORE()                               \
  _dispatch_unhandled /*verifier should prevent this*/ : {                  \
    IREE_ASSERT(0);                                                         \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unhandled opcode"); \
  }
#define IREE_VM_ISA_DISPATCH_UNHANDLED_PREFIX(op_name, ext)        \
  _dispatch_CORE_##op_name : {                                     \
    IREE_ASSERT(0);                                                \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,             \
                            "unhandled dispatch extension " #ext); \
  }

#define IREE_VM_ISA_DISPATCH_OP(ext, op_name, body)                       \
  _dispatch_##ext##_##op_name :;                                          \
  IREE_DISPATCH_TRACE_INSTRUCTION(IREE_VM_ISA_PC_OFFSET_##ext, #op_name); \
  body;                                                                   \
  goto* kDispatchTable_CORE[bytecode_data[pc++]];

#define IREE_VM_ISA_DISPATCH_BEGIN_PREFIX(op_name, ext)                       \
  _dispatch_CORE_##op_name : goto* kDispatchTable_##ext[bytecode_data[pc++]]; \
  while (1)
#define IREE_VM_ISA_DISPATCH_END_PREFIX() \
  goto* kDispatchTable_CORE[bytecode_data[pc++]];

#else

// Switch-based dispatch. This is strictly less efficient than the computed
// goto approach above but is universally supported.

#define IREE_VM_ISA_DISPATCH_BEGIN_CORE() \
  while (1) {                             \
    switch (bytecode_data[pc++])
#define IREE_VM_ISA_DISPATCH_END_CORE() }

#define IREE_VM_ISA_DISPATCH_DEFINE_TABLES()

#define IREE_VM_ISA_DISPATCH_UNHANDLED_CORE()             \
  default: {                                              \
    IREE_ASSERT(0);                                       \
    IREE_BUILTIN_UNREACHABLE(); /* ok because verified */ \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,    \
                            "unhandled core opcode");     \
  }
#define IREE_VM_ISA_DISPATCH_UNHANDLED_PREFIX(op_name, ext)        \
  case IREE_VM_OP_CORE_##op_name: {                                \
    IREE_ASSERT(0);                                                \
    IREE_BUILTIN_UNREACHABLE(); /* ok because verified */          \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,             \
                            "unhandled dispatch extension " #ext); \
  }

#define IREE_VM_ISA_DISPATCH_OP(ext, op_name, body)                         \
  case IREE_VM_OP_##ext##_##op_name: {                                      \
    IREE_DISPATCH_TRACE_INSTRUCTION(IREE_VM_ISA_PC_OFFSET_##ext, #op_name); \
    body;                                                                   \
  } break;

#define IREE_VM_ISA_DISPATCH_BEGIN_PREFIX(op_name, ext) \
  case IREE_VM_OP_CORE_##op_name: {                     \
    switch (bytecode_data[pc++])
#define IREE_VM_ISA_DISPATCH_END_PREFIX() \
  break;                                  \
  }

#endif  // IREE_DISPATCH_MODE_COMPUTED_GOTO

// Common dispatch op macros

#define IREE_VM_ISA_DISPATCH_OP_CORE_UNARY_I32(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(CORE, op_name, {                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(operand);            \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_I32(result);              \
    *result = op_func(operand);                                  \
  });

#define IREE_VM_ISA_DISPATCH_OP_CORE_UNARY_I64(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(CORE, op_name, {                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(operand);            \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_I64(result);              \
    *result = op_func(operand);                                  \
  });

#define IREE_VM_ISA_DISPATCH_OP_CORE_BINARY_I32(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(CORE, op_name, {                        \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(lhs);                 \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(rhs);                 \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_I32(result);               \
    *result = op_func(lhs, rhs);                                  \
  });

#define IREE_VM_ISA_DISPATCH_OP_CORE_BINARY_I64(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(CORE, op_name, {                        \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(lhs);                 \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(rhs);                 \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_I64(result);               \
    *result = op_func(lhs, rhs);                                  \
  });

#define IREE_VM_ISA_DISPATCH_OP_CORE_TERNARY_I32(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(CORE, op_name, {                         \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(a);                    \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(b);                    \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I32(c);                    \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_I32(result);                \
    *result = op_func(a, b, c);                                    \
  });

#define IREE_VM_ISA_DISPATCH_OP_CORE_TERNARY_I64(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(CORE, op_name, {                         \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(a);                    \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(b);                    \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_I64(c);                    \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_I64(result);                \
    *result = op_func(a, b, c);                                    \
  });

#define IREE_VM_ISA_DISPATCH_OP_EXT_F32_UNARY_F32(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(EXT_F32, op_name, {                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(operand);               \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_F32(result);                 \
    *result = op_func(operand);                                     \
  });

#define IREE_VM_ISA_DISPATCH_OP_EXT_F32_BINARY_F32(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(EXT_F32, op_name, {                        \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(lhs);                    \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(rhs);                    \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_F32(result);                  \
    *result = op_func(lhs, rhs);                                     \
  });

#define IREE_VM_ISA_DISPATCH_OP_EXT_F32_TERNARY_F32(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(EXT_F32, op_name, {                         \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(a);                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(b);                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F32(c);                       \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_F32(result);                   \
    *result = op_func(a, b, c);                                       \
  });

#define IREE_VM_ISA_DISPATCH_OP_EXT_F64_UNARY_F64(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(EXT_F64, op_name, {                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(operand);               \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_F64(result);                 \
    *result = op_func(operand);                                     \
  });

#define IREE_VM_ISA_DISPATCH_OP_EXT_F64_BINARY_F64(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(EXT_F64, op_name, {                        \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(lhs);                    \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(rhs);                    \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_F64(result);                  \
    *result = op_func(lhs, rhs);                                     \
  });

#define IREE_VM_ISA_DISPATCH_OP_EXT_F64_TERNARY_F64(op_name, op_func) \
  IREE_VM_ISA_DISPATCH_OP(EXT_F64, op_name, {                         \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(a);                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(b);                       \
    IREE_VM_ISA_DISPATCH_DECODE_OPERAND_F64(c);                       \
    IREE_VM_ISA_DISPATCH_DECODE_RESULT_F64(result);                   \
    *result = op_func(a, b, c);                                       \
  });

#endif  // IREE_VM_BYTECODE_DISPATCH_UTIL_H_
