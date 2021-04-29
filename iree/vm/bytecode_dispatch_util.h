// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_VM_BYTECODE_DISPATCH_UTIL_H_
#define IREE_VM_BYTECODE_DISPATCH_UTIL_H_

#include <assert.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/config.h"
#include "iree/base/target_platform.h"
#include "iree/vm/bytecode_module_impl.h"
#include "iree/vm/generated/bytecode_op_table.h"

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
typedef struct {
  // Ordinal mask defining which ordinal bits are valid. All i32 indexing must
  // be ANDed with this mask.
  uint16_t i32_mask;
  // 16-byte aligned i32 register array.
  int32_t* i32;
  // Ordinal mask defining which ordinal bits are valid. All ref indexing must
  // be ANDed with this mask.
  uint16_t ref_mask;
  // Naturally aligned ref register array.
  iree_vm_ref_t* ref;
} iree_vm_registers_t;

// Storage associated with each stack frame of a bytecode function.
// NOTE: we cannot store pointers to the stack in here as the stack may be
// reallocated.
typedef struct {
  // Pointer to a register list within the stack frame where return registers
  // will be stored by callees upon return.
  const iree_vm_register_list_t* return_registers;

  // Counts of each register type rounded up to the next power of two.
  iree_host_size_t i32_register_count;
  iree_host_size_t ref_register_count;

  // Relative byte offsets from the head of this struct.
  iree_host_size_t i32_register_offset;
  iree_host_size_t ref_register_offset;
} iree_vm_bytecode_frame_storage_t;

// Interleaved src-dst register sets for branch register remapping.
// This structure is an overlay for the bytecode that is serialized in a
// matching format.
typedef struct {
  uint16_t size;
  struct pair {
    uint16_t src_reg;
    uint16_t dst_reg;
  } pairs[];
} iree_vm_register_remap_list_t;
static_assert(iree_alignof(iree_vm_register_remap_list_t) == 2,
              "Expecting byte alignment (to avoid padding)");
static_assert(offsetof(iree_vm_register_remap_list_t, pairs) == 2,
              "Expect no padding in the struct");

// Maps a type ID to a type def with clamping for out of bounds values.
static inline const iree_vm_type_def_t* iree_vm_map_type(
    iree_vm_bytecode_module_t* module, int32_t type_id) {
  type_id = type_id >= module->type_count ? 0 : type_id;
  return &module->type_table[type_id];
}

//===----------------------------------------------------------------------===//
// Debugging utilities
//===----------------------------------------------------------------------===//

// Enable to get some verbose logging; better than nothing until we have some
// better tooling.
#define IREE_DISPATCH_LOGGING 0

#if IREE_DISPATCH_LOGGING
#include <stdio.h>
#define IREE_DISPATCH_LOG_OPCODE(op_name) \
  fprintf(stderr, "DISPATCH %d %s\n", (int)pc, op_name)
#define IREE_DISPATCH_LOG_CALL(target_function) \
  fprintf(stderr, "CALL -> %s\n", iree_vm_function_name(target_function).data);
#else
#define IREE_DISPATCH_LOG_OPCODE(...)
#define IREE_DISPATCH_LOG_CALL(...)
#endif  // IREE_DISPATCH_LOGGING

#if defined(IREE_COMPILER_MSVC) && !defined(IREE_COMPILER_CLANG)
#define IREE_DISPATCH_MODE_SWITCH 1
#else
#define IREE_DISPATCH_MODE_COMPUTED_GOTO 1
#endif  // MSVC

#ifndef NDEBUG
#define VMCHECK(expr) assert(expr)
#else
#define VMCHECK(expr)
#endif  // NDEBUG

//===----------------------------------------------------------------------===//
// Bytecode data reading with little-/big-endian support
//===----------------------------------------------------------------------===//

static const int kRegSize = sizeof(uint16_t);

// Bytecode data access macros for reading values of a given type from a byte
// offset within the current function.
#if defined(IREE_ENDIANNESS_LITTLE)
#define OP_I8(i) bytecode_data[pc + (i)]
#define OP_I16(i) *((uint16_t*)&bytecode_data[pc + (i)])
#define OP_I32(i) *((uint32_t*)&bytecode_data[pc + (i)])
#define OP_I64(i) *((uint64_t*)&bytecode_data[pc + (i)])
#else
#define OP_I8(i) bytecode_data[pc + (i)]
#define OP_I16(i)                           \
  ((uint16_t)bytecode_data[pc + 0 + (i)]) | \
      ((uint16_t)bytecode_data[pc + 1 + (i)] << 8)
#define OP_I32(i)                                     \
  ((uint32_t)bytecode_data[pc + 0 + (i)]) |           \
      ((uint32_t)bytecode_data[pc + 1 + (i)] << 8) |  \
      ((uint32_t)bytecode_data[pc + 2 + (i)] << 16) | \
      ((uint32_t)bytecode_data[pc + 3 + (i)] << 24)
#define OP_I64(i)                                     \
  ((uint64_t)bytecode_data[pc + 0 + (i)]) |           \
      ((uint64_t)bytecode_data[pc + 1 + (i)] << 8) |  \
      ((uint64_t)bytecode_data[pc + 2 + (i)] << 16) | \
      ((uint64_t)bytecode_data[pc + 3 + (i)] << 24) | \
      ((uint64_t)bytecode_data[pc + 4 + (i)] << 32) | \
      ((uint64_t)bytecode_data[pc + 5 + (i)] << 40) | \
      ((uint64_t)bytecode_data[pc + 6 + (i)] << 48) | \
      ((uint64_t)bytecode_data[pc + 7 + (i)] << 56)
#endif  // IREE_ENDIANNESS_LITTLE

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
#define VM_DecOpcode(opcode) VM_DecConstI8(#opcode)
#define VM_DecFuncAttr(name) VM_DecConstI32(name)
#define VM_DecGlobalAttr(name) VM_DecConstI32(name)
#define VM_DecRodataAttr(name) VM_DecConstI32(name)
#define VM_DecType(name)               \
  iree_vm_map_type(module, OP_I32(0)); \
  pc += 4;
#define VM_DecTypeOf(name) VM_DecType(name)
#define VM_DecIntAttr32(name) VM_DecConstI32(name)
#define VM_DecIntAttr64(name) VM_DecConstI64(name)
#define VM_DecStrAttr(name, out_str)                     \
  (out_str)->size = (iree_host_size_t)OP_I16(0);         \
  (out_str)->data = (const char*)&bytecode_data[pc + 2]; \
  pc += 2 + (out_str)->size;
#define VM_DecBranchTarget(block_name) VM_DecConstI32(name)
#define VM_DecBranchOperands(operands_name)                                   \
  (const iree_vm_register_remap_list_t*)&bytecode_data[pc];                   \
  pc +=                                                                       \
      kRegSize + ((const iree_vm_register_list_t*)&bytecode_data[pc])->size * \
                     2 * kRegSize;
#define VM_DecOperandRegI32(name)      \
  regs.i32[OP_I16(0) & regs.i32_mask]; \
  pc += kRegSize;
#define VM_DecOperandRegI64(name)                           \
  *((int64_t*)&regs.i32[OP_I16(0) & (regs.i32_mask & ~1)]); \
  pc += kRegSize;
#define VM_DecOperandRegRef(name, out_is_move)             \
  &regs.ref[OP_I16(0) & regs.ref_mask];                    \
  *(out_is_move) = OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT; \
  pc += kRegSize;
#define VM_DecVariadicOperands(name)                  \
  (const iree_vm_register_list_t*)&bytecode_data[pc]; \
  pc += kRegSize +                                    \
        ((const iree_vm_register_list_t*)&bytecode_data[pc])->size * kRegSize;
#define VM_DecResultRegI32(name)        \
  &regs.i32[OP_I16(0) & regs.i32_mask]; \
  pc += kRegSize;
#define VM_DecResultRegI64(name)                           \
  ((int64_t*)&regs.i32[OP_I16(0) & (regs.i32_mask & ~1)]); \
  pc += kRegSize;
#define VM_DecResultRegRef(name, out_is_move)              \
  &regs.ref[OP_I16(0) & regs.ref_mask];                    \
  *(out_is_move) = OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT; \
  pc += kRegSize;
#define VM_DecVariadicResults(name) VM_DecVariadicOperands(name)

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
#if IREE_VM_EXT_I64_ENABLE
#define DECLARE_DISPATCH_EXT_I64_OPC(ordinal, name) &&_dispatch_EXT_I64_##name,
#define DEFINE_DISPATCH_TABLE_EXT_I64()                                       \
  static const void* kDispatchTable_EXT_I64[256] = {IREE_VM_OP_EXT_I64_TABLE( \
      DECLARE_DISPATCH_EXT_I64_OPC, DECLARE_DISPATCH_EXT_RSV)};
#else
#define DEFINE_DISPATCH_TABLE_EXT_I64()
#endif  // IREE_VM_EXT_I64_ENABLE

#define DEFINE_DISPATCH_TABLES() \
  DEFINE_DISPATCH_TABLE_CORE();  \
  DEFINE_DISPATCH_TABLE_EXT_I64();

#define DISPATCH_UNHANDLED_CORE()                                           \
  _dispatch_unhandled : {                                                   \
    VMCHECK(0);                                                             \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unhandled opcode"); \
  }
#define DISPATCH_UNHANDLED_EXT()

#define DISPATCH_OP(ext, op_name, body)                             \
  _dispatch_##ext##_##op_name : IREE_DISPATCH_LOG_OPCODE(#op_name); \
  body;                                                             \
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

#define DISPATCH_UNHANDLED_CORE()                      \
  default: {                                           \
    VMCHECK(0);                                        \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, \
                            "unhandled core opcode");  \
  }
#define DISPATCH_UNHANDLED_EXT                             \
  () default : {                                           \
    VMCHECK(0);                                            \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,     \
                            "unhandled extension opcode"); \
  }

#define DISPATCH_OP(ext, op_name, body) \
  case IREE_VM_OP_##ext##_##op_name: {  \
    IREE_DISPATCH_LOG_OPCODE(#op_name); \
    body;                               \
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

#define DISPATCH_OP_CORE_BINARY_I32(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                        \
    int32_t lhs = VM_DecOperandRegI32("lhs");         \
    int32_t rhs = VM_DecOperandRegI32("rhs");         \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = op_func(lhs, rhs);                      \
  });

#define DISPATCH_OP_EXT_I64_UNARY_I64(op_name, op_func) \
  DISPATCH_OP(EXT_I64, op_name, {                       \
    int64_t operand = VM_DecOperandRegI64("operand");   \
    int64_t* result = VM_DecResultRegI64("result");     \
    *result = op_func(operand);                         \
  });

#define DISPATCH_OP_EXT_I64_BINARY_I64(op_name, op_func) \
  DISPATCH_OP(EXT_I64, op_name, {                        \
    int64_t lhs = VM_DecOperandRegI64("lhs");            \
    int64_t rhs = VM_DecOperandRegI64("rhs");            \
    int64_t* result = VM_DecResultRegI64("result");      \
    *result = op_func(lhs, rhs);                         \
  });

#endif  // IREE_VM_BYTECODE_DISPATCH_UTIL_H_
