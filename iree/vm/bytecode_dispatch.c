// Copyright 2019 Google LLC
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

#include <assert.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/target_platform.h"
#include "iree/vm/bytecode_module_impl.h"
#include "iree/vm/bytecode_op_table.h"
#include "iree/vm/list.h"

// Enable to get some verbose logging; better than nothing until we have some
// better tooling.
#define IREE_DISPATCH_LOGGING 0

#if IREE_DISPATCH_LOGGING
#include <stdio.h>
#define IREE_DISPATCH_LOG_OPCODE(op_name) \
  fprintf(stderr, "DISPATCH %d %s\n", (int)pc, op_name)
#define IREE_DISPATCH_LOG_CALL(target_function) \
  fprintf(stderr, "CALL -> %s\n", iree_vm_function_name(&target_function).data);
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

// Remaps registers from a source set to a destination set within the same stack
// frame. This is a way to perform a conditional multi-mov sequence instead of
// requiring the additional bytecode representation of the conditional movs.
//
// This assumes that the remapping list is properly ordered such that there are
// no swapping hazards (such as 0->1,1->0). The register allocator in the
// compiler should ensure this is the case when it can occur.
static void iree_vm_bytecode_dispatch_remap_branch_registers(
    const iree_vm_registers_t regs,
    const iree_vm_register_remap_list_t* remap_list) {
  for (int i = 0; i < remap_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = remap_list->pairs[i].src_reg;
    uint16_t dst_reg = remap_list->pairs[i].dst_reg;
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(src_reg & IREE_REF_REGISTER_MOVE_BIT,
                                 &regs.ref[src_reg & regs.ref_mask],
                                 &regs.ref[dst_reg & regs.ref_mask]);
    } else {
      regs.i32[dst_reg & regs.i32_mask] = regs.i32[src_reg & regs.i32_mask];
    }
  }
}

// Discards ref registers in the list if they are marked move.
// This can be used to eagerly release resources we don't need and reduces
// memory consumption if used effectively prior to yields/waits.
static void iree_vm_bytecode_dispatch_discard_registers(
    const iree_vm_registers_t regs, const iree_vm_register_list_t* reg_list) {
  for (int i = 0; i < reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    uint16_t reg = reg_list->registers[i];
    if ((reg & (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) ==
        (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) {
      iree_vm_ref_release(&regs.ref[reg & regs.ref_mask]);
    }
  }
}

static const int kRegSize = sizeof(uint16_t);

// Bytecode data access macros for reading values of a given type from a byte
// offset within the current function.
#if defined(IREE_IS_LITTLE_ENDIAN)
#define OP_I8(i) bytecode_data[pc + i]
#define OP_I16(i) *((uint16_t*)&bytecode_data[pc + i])
#define OP_I32(i) *((uint32_t*)&bytecode_data[pc + i])
#else
#define OP_I8(i) bytecode_data[pc + i]
#define OP_I16(i)                         \
  ((uint16_t)bytecode_data[pc + 0 + i]) | \
      ((uint16_t)bytecode_data[pc + 1 + i] << 8)
#define OP_I32(i)                                   \
  ((uint32_t)bytecode_data[pc + 0 + i]) |           \
      ((uint32_t)bytecode_data[pc + 1 + i] << 8) |  \
      ((uint32_t)bytecode_data[pc + 2 + i] << 16) | \
      ((uint32_t)bytecode_data[pc + 3 + i] << 24)
#endif  // IREE_IS_LITTLE_ENDIAN

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
#define VM_DecOpcode(opcode) VM_DecConstI8(#opcode)
#define VM_DecFuncAttr(name) VM_DecConstI32(name)
#define VM_DecGlobalAttr(name) VM_DecConstI32(name)
#define VM_DecRodataAttr(name) VM_DecConstI32(name)
#define VM_DecType(name)               \
  iree_vm_map_type(module, OP_I32(0)); \
  pc += 4;
#define VM_DecTypeOf(name) VM_DecType(name)
#define VM_DecIntAttr32(name) VM_DecConstI32(name)
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
#define VM_DecResultRegRef(name, out_is_move)              \
  &regs.ref[OP_I16(0) & regs.ref_mask];                    \
  *(out_is_move) = OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT; \
  pc += kRegSize;
#define VM_DecVariadicResults(name) VM_DecVariadicOperands(name)

iree_status_t iree_vm_bytecode_dispatch(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, iree_vm_stack_t* stack,
    iree_vm_stack_frame_t* entry_frame,
    iree_vm_execution_result_t* out_result) {
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
#define BEGIN_DISPATCH()                     \
  goto* kDispatchTable[bytecode_data[pc++]]; \
  while (1)

#define END_DISPATCH()

#define DECLARE_DISPATCH_OPC(ordinal, name) &&_dispatch_##name,
#define DECLARE_DISPATCH_RSV(ordinal) &&_dispatch_unhandled,
  static const void* kDispatchTable[256] = {
      IREE_VM_OP_CORE_TABLE(DECLARE_DISPATCH_OPC, DECLARE_DISPATCH_RSV)};

#define DISPATCH_UNHANDLED() \
  _dispatch_unhandled:       \
  VMCHECK(0);                \
  return IREE_STATUS_UNIMPLEMENTED;

#define DISPATCH_OP(op_name, body)                          \
  _dispatch_##op_name : IREE_DISPATCH_LOG_OPCODE(#op_name); \
  body;                                                     \
  goto* kDispatchTable[bytecode_data[pc++]];

#else

  // Switch-based dispatch. This is strictly less efficient than the computed
  // goto approach above but is universally supported.

#define BEGIN_DISPATCH() \
  while (1) {            \
    switch (bytecode_data[pc++])

#define END_DISPATCH() }

#define DISPATCH_UNHANDLED() \
  default:                   \
    VMCHECK(0);              \
    return IREE_STATUS_UNIMPLEMENTED;

#define DISPATCH_OP(op_name, body)      \
  case IREE_VM_OP_CORE_##op_name:       \
    IREE_DISPATCH_LOG_OPCODE(#op_name); \
    body;                               \
    break;

#endif  // IREE_DISPATCH_MODE_COMPUTED_GOTO

  // Primary dispatch state. This is our 'native stack frame' and really
  // just enough to make dereferencing common addresses (like the current
  // offset) faster. You can think of this like CPU state (like PC).
  //
  // The hope is that the compiler decides to keep these in registers (as
  // they are touched for every instruction executed). The frame will change
  // as we call into different functions.
  iree_vm_stack_frame_t* current_frame = entry_frame;
  iree_vm_registers_t regs = entry_frame->registers;
  const uint8_t* bytecode_data =
      module->bytecode_data.data +
      module->function_descriptor_table[current_frame->function.ordinal]
          .bytecode_offset;
  iree_vm_source_offset_t pc = current_frame->pc;
  const int32_t entry_frame_depth = entry_frame->depth;

  memset(out_result, 0, sizeof(*out_result));

  BEGIN_DISPATCH() {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    DISPATCH_OP(GlobalLoadI32, {
      int32_t byte_offset = VM_DecGlobalAttr("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int32_t* value = VM_DecResultRegI32("value");
      const int32_t* global_ptr =
          (const int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *value = *global_ptr;
    });

    DISPATCH_OP(GlobalStoreI32, {
      int32_t byte_offset = VM_DecGlobalAttr("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int32_t value = VM_DecOperandRegI32("value");
      int32_t* global_ptr =
          (int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *global_ptr = value;
    });

    DISPATCH_OP(GlobalLoadIndirectI32, {
      int32_t byte_offset = VM_DecOperandRegI32("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int32_t* value = VM_DecResultRegI32("value");
      const int32_t* global_ptr =
          (const int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *value = *global_ptr;
    });

    DISPATCH_OP(GlobalStoreIndirectI32, {
      int32_t byte_offset = VM_DecOperandRegI32("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int32_t value = VM_DecOperandRegI32("value");
      int32_t* global_ptr =
          (int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *global_ptr = value;
    });

    DISPATCH_OP(GlobalLoadRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(result_is_move, global_ref,
                                         type_def->ref_type, result);
    });

    DISPATCH_OP(GlobalStoreRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool value_is_move;
      iree_vm_ref_t* value = VM_DecOperandRegRef("value", &value_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(value_is_move, value,
                                         type_def->ref_type, global_ref);
    });

    DISPATCH_OP(GlobalLoadIndirectRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(result_is_move, global_ref,
                                         type_def->ref_type, result);
    });

    DISPATCH_OP(GlobalStoreIndirectRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool value_is_move;
      iree_vm_ref_t* value = VM_DecOperandRegRef("value", &value_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(value_is_move, value,
                                         type_def->ref_type, global_ref);
    });

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    DISPATCH_OP(ConstI32, {
      int32_t value = VM_DecIntAttr32("value");
      int32_t* result = VM_DecResultRegI32("result");
      *result = value;
    });

    DISPATCH_OP(ConstI32Zero, {
      int32_t* result = VM_DecResultRegI32("result");
      *result = 0;
    });

    DISPATCH_OP(ConstRefZero, {
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_ref_release(result);
    });

    DISPATCH_OP(ConstRefRodata, {
      int32_t rodata_ordinal = VM_DecRodataAttr("rodata");
      if (rodata_ordinal < 0 ||
          rodata_ordinal >= module_state->rodata_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      iree_vm_ref_wrap_retain(&module_state->rodata_ref_table[rodata_ordinal],
                              iree_vm_ro_byte_buffer_type_id(), result);
    });

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    DISPATCH_OP(ListAlloc, {
      const iree_vm_type_def_t* element_type_def = VM_DecTypeOf("element_type");
      iree_host_size_t initial_capacity =
          VM_DecOperandRegI32("initial_capacity");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_list_t* list = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_list_create(
          element_type_def, initial_capacity, module_state->allocator, &list));
      iree_vm_ref_wrap_assign(list, iree_vm_list_type_id(), result);
    });

    DISPATCH_OP(ListReserve, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      int32_t minimum_capacity = VM_DecOperandRegI32("minimum_capacity");
      IREE_RETURN_IF_ERROR(iree_vm_list_reserve(list, minimum_capacity));
    });

    DISPATCH_OP(ListSize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      int32_t* result = VM_DecResultRegI32("result");
      *result = (int32_t)iree_vm_list_size(list);
    });

    DISPATCH_OP(ListResize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      int32_t new_size = VM_DecOperandRegI32("new_size");
      IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, new_size));
    });

    DISPATCH_OP(ListGetI32, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      int32_t index = VM_DecOperandRegI32("index");
      int32_t* result = VM_DecResultRegI32("result");
      iree_vm_value_t value;
      IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
          list, index, IREE_VM_VALUE_TYPE_I32, &value));
      *result = value.i32;
    });

    DISPATCH_OP(ListSetI32, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      int32_t index = VM_DecOperandRegI32("index");
      int32_t raw_value = VM_DecOperandRegI32("raw_value");
      iree_vm_value_t value = iree_vm_value_make_i32(raw_value);
      IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
    });

    DISPATCH_OP(ListGetRef, {
      // bool list_is_move;
      // iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      // iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      // if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      // int32_t index = VM_DecOperandRegI32("index");
      // iree_vm_ref_t* result = VM_DecResultRegRef("result");
      return IREE_STATUS_UNIMPLEMENTED;
    });

    DISPATCH_OP(ListSetRef, {
      // bool list_is_move;
      // iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      // iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      // if (!list) return IREE_STATUS_INVALID_ARGUMENT;
      // int32_t index = VM_DecOperandRegI32("index");
      // bool operand_is_move = VM_DecOperandRegRefIsMove("value");
      // iree_vm_ref_t* operand = VM_DecOperandRegRef("value");
      return IREE_STATUS_UNIMPLEMENTED;
    });

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    DISPATCH_OP(SelectI32, {
      int32_t condition = VM_DecOperandRegI32("condition");
      int32_t true_value = VM_DecOperandRegI32("true_value");
      int32_t false_value = VM_DecOperandRegI32("false_value");
      int32_t* result = VM_DecResultRegI32("result");
      *result = condition ? true_value : false_value;
    });

    DISPATCH_OP(SelectRef, {
      int32_t condition = VM_DecOperandRegI32("condition");
      // TODO(benvanik): remove the type_id and use either LHS/RHS (if both are
      // null then output is always null so no need to know the type).
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("true_value");
      bool true_value_is_move;
      iree_vm_ref_t* true_value =
          VM_DecOperandRegRef("true_value", &true_value_is_move);
      bool false_value_is_move;
      iree_vm_ref_t* false_value =
          VM_DecOperandRegRef("false_value", &false_value_is_move);
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      if (condition) {
        // Select LHS.
        iree_vm_ref_retain_or_move_checked(true_value_is_move, true_value,
                                           type_def->ref_type, result);
        if (false_value_is_move) iree_vm_ref_release(false_value);
      } else {
        // Select RHS.
        iree_vm_ref_retain_or_move_checked(false_value_is_move, false_value,
                                           type_def->ref_type, result);
        if (true_value_is_move) iree_vm_ref_release(true_value);
      }
    });

    DISPATCH_OP(SwitchI32, {
      int32_t index = VM_DecOperandRegI32("index");
      int32_t default_value = VM_DecIntAttr32("default_value");
      const iree_vm_register_list_t* value_reg_list =
          VM_DecVariadicOperands("values");
      int32_t* result = VM_DecResultRegI32("result");
      if (index >= 0 && index < value_reg_list->size) {
        *result = regs.i32[value_reg_list->registers[index] & regs.i32_mask];
      } else {
        *result = default_value;
      }
    });

    DISPATCH_OP(SwitchRef, {
      int32_t index = VM_DecOperandRegI32("index");
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("result");
      bool default_is_move;
      iree_vm_ref_t* default_value =
          VM_DecOperandRegRef("default_value", &default_is_move);
      const iree_vm_register_list_t* value_reg_list =
          VM_DecVariadicOperands("values");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      if (index >= 0 && index < value_reg_list->size) {
        bool is_move =
            value_reg_list->registers[index] & IREE_REF_REGISTER_MOVE_BIT;
        iree_vm_ref_t* new_value =
            &regs.ref[value_reg_list->registers[index] & regs.ref_mask];
        iree_vm_ref_retain_or_move_checked(is_move, new_value,
                                           type_def->ref_type, result);
      } else {
        iree_vm_ref_retain_or_move_checked(default_is_move, default_value,
                                           type_def->ref_type, result);
      }
    });

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_UNARY_ALU_I32(op_name, type, op)  \
  DISPATCH_OP(op_name, {                              \
    int32_t operand = VM_DecOperandRegI32("operand"); \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = (int32_t)(op((type)operand));           \
  });

#define DISPATCH_OP_BINARY_ALU_I32(op_name, type, op) \
  DISPATCH_OP(op_name, {                              \
    int32_t lhs = VM_DecOperandRegI32("lhs");         \
    int32_t rhs = VM_DecOperandRegI32("rhs");         \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = (int32_t)(((type)lhs)op((type)rhs));    \
  });

    DISPATCH_OP_BINARY_ALU_I32(AddI32, int32_t, +);
    DISPATCH_OP_BINARY_ALU_I32(SubI32, int32_t, -);
    DISPATCH_OP_BINARY_ALU_I32(MulI32, int32_t, *);
    DISPATCH_OP_BINARY_ALU_I32(DivI32S, int32_t, /);
    DISPATCH_OP_BINARY_ALU_I32(DivI32U, uint32_t, /);
    DISPATCH_OP_BINARY_ALU_I32(RemI32S, int32_t, %);
    DISPATCH_OP_BINARY_ALU_I32(RemI32U, uint32_t, %);
    DISPATCH_OP_UNARY_ALU_I32(NotI32, uint32_t, ~);
    DISPATCH_OP_BINARY_ALU_I32(AndI32, uint32_t, &);
    DISPATCH_OP_BINARY_ALU_I32(OrI32, uint32_t, |);
    DISPATCH_OP_BINARY_ALU_I32(XorI32, uint32_t, ^);

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CAST_I32(op_name, src_type, dst_type) \
  DISPATCH_OP(op_name, {                                  \
    int32_t operand = VM_DecOperandRegI32("operand");     \
    int32_t* result = VM_DecResultRegI32("result");       \
    *result = (dst_type)((src_type)operand);              \
  });

    DISPATCH_OP_CAST_I32(TruncI32I8, uint8_t, uint32_t);
    DISPATCH_OP_CAST_I32(TruncI32I16, uint16_t, uint32_t);
    DISPATCH_OP_CAST_I32(ExtI8I32S, int8_t, int32_t);
    DISPATCH_OP_CAST_I32(ExtI8I32U, uint8_t, uint32_t);
    DISPATCH_OP_CAST_I32(ExtI16I32S, int16_t, int32_t);
    DISPATCH_OP_CAST_I32(ExtI16I32U, uint16_t, uint32_t);

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_SHIFT_I32(op_name, type, op)      \
  DISPATCH_OP(op_name, {                              \
    int32_t operand = VM_DecOperandRegI32("operand"); \
    int32_t amount = VM_DecConstI8("amount");         \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = (int32_t)(((type)operand)op amount);    \
  });

    DISPATCH_OP_SHIFT_I32(ShlI32, int32_t, <<);
    DISPATCH_OP_SHIFT_I32(ShrI32S, int32_t, >>);
    DISPATCH_OP_SHIFT_I32(ShrI32U, uint32_t, >>);

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CMP_I32(op_name, type, op)      \
  DISPATCH_OP(op_name, {                            \
    int32_t lhs = VM_DecOperandRegI32("lhs");       \
    int32_t rhs = VM_DecOperandRegI32("rhs");       \
    int32_t* result = VM_DecResultRegI32("result"); \
    *result = (((type)lhs)op((type)rhs)) ? 1 : 0;   \
  });

    DISPATCH_OP_CMP_I32(CmpEQI32, int32_t, ==);
    DISPATCH_OP_CMP_I32(CmpNEI32, int32_t, !=);
    DISPATCH_OP_CMP_I32(CmpLTI32S, int32_t, <);
    DISPATCH_OP_CMP_I32(CmpLTI32U, uint32_t, <);
    DISPATCH_OP(CmpNZI32, {
      int32_t operand = VM_DecOperandRegI32("operand");
      int32_t* result = VM_DecResultRegI32("result");
      *result = (operand != 0) ? 1 : 0;
    });

    DISPATCH_OP(CmpEQRef, {
      bool lhs_is_move;
      iree_vm_ref_t* lhs = VM_DecOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      iree_vm_ref_t* rhs = VM_DecOperandRegRef("rhs", &rhs_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = iree_vm_ref_equal(lhs, rhs);
      if (lhs_is_move) iree_vm_ref_release(lhs);
      if (rhs_is_move) iree_vm_ref_release(rhs);
    });
    DISPATCH_OP(CmpNERef, {
      bool lhs_is_move;
      iree_vm_ref_t* lhs = VM_DecOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      iree_vm_ref_t* rhs = VM_DecOperandRegRef("rhs", &rhs_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = !iree_vm_ref_equal(lhs, rhs);
      if (lhs_is_move) iree_vm_ref_release(lhs);
      if (rhs_is_move) iree_vm_ref_release(rhs);
    });
    DISPATCH_OP(CmpNZRef, {
      bool operand_is_move;
      iree_vm_ref_t* operand = VM_DecOperandRegRef("operand", &operand_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = operand->ptr != NULL ? 1 : 0;
      if (operand_is_move) iree_vm_ref_release(operand);
    });

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    DISPATCH_OP(Branch, {
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      pc = block_pc;
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
    });

    DISPATCH_OP(CondBranch, {
      int32_t condition = VM_DecOperandRegI32("condition");
      int32_t true_block_pc = VM_DecBranchTarget("true_dest");
      const iree_vm_register_remap_list_t* true_remap_list =
          VM_DecBranchOperands("true_operands");
      int32_t false_block_pc = VM_DecBranchTarget("false_dest");
      const iree_vm_register_remap_list_t* false_remap_list =
          VM_DecBranchOperands("false_operands");
      if (condition) {
        pc = true_block_pc;
        iree_vm_bytecode_dispatch_remap_branch_registers(regs, true_remap_list);
      } else {
        pc = false_block_pc;
        iree_vm_bytecode_dispatch_remap_branch_registers(regs,
                                                         false_remap_list);
      }
    });

    DISPATCH_OP(Call, {
      int32_t function_ordinal = VM_DecFuncAttr("callee");
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      const iree_vm_register_list_t* dst_reg_list =
          VM_DecVariadicResults("results");
      current_frame->pc = pc;

      // NOTE: we assume validation has ensured these functions exist.
      // TODO(benvanik): something more clever than just a high bit?
      int is_import = (function_ordinal & 0x80000000u) != 0;
      if (is_import) {
        // Call external function.
        iree_vm_function_call_t call;
        memset(&call, 0, sizeof(call));
        call.function =
            module_state->import_table[function_ordinal & 0x7FFFFFFFu];
        call.argument_registers = src_reg_list;
        call.result_registers = dst_reg_list;
        IREE_DISPATCH_LOG_CALL(call.function);
        iree_status_t call_status = call.function.module->begin_call(
            call.function.module->self, stack, &call, out_result);
        if (!iree_status_is_ok(call_status)) {
          // TODO(benvanik): set execution result to failure/capture stack.
          return call_status;
        }
      } else {
        // Switch execution to the target function and continue running in the
        // bytecode dispatcher.
        iree_vm_function_t target_function;
        target_function.module = &module->interface;
        target_function.linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
        target_function.ordinal = function_ordinal;
        const iree_vm_FunctionDescriptor_t* target_descriptor =
            &module->function_descriptor_table[function_ordinal];
        target_function.i32_register_count =
            target_descriptor->i32_register_count;
        target_function.ref_register_count =
            target_descriptor->ref_register_count;
        IREE_DISPATCH_LOG_CALL(target_function);
        iree_status_t enter_status = iree_vm_stack_function_enter(
            stack, target_function, src_reg_list, dst_reg_list, &current_frame);
        if (!iree_status_is_ok(enter_status)) {
          // TODO(benvanik): set execution result to stack overflow.
          return enter_status;
        }
        regs = current_frame->registers;
        bytecode_data =
            module->bytecode_data.data + target_descriptor->bytecode_offset;
        pc = current_frame->pc;
      }
    });

    DISPATCH_OP(CallVariadic, {
      // TODO(benvanik): dedupe with above or merge and always have the seg size
      // list be present (but empty) for non-variadic calls.
      int32_t function_ordinal = VM_DecFuncAttr("callee");
      const iree_vm_register_list_t* seg_size_list =
          VM_DecVariadicOperands("segment_sizes");
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      const iree_vm_register_list_t* dst_reg_list =
          VM_DecVariadicResults("results");
      current_frame->pc = pc;

      // NOTE: we assume validation has ensured these functions exist.
      // TODO(benvanik): something more clever than just a high bit?
      int is_import = (function_ordinal & 0x80000000u) != 0;
      if (!is_import) {
        // Variadic calls are currently only supported for import functions.
        return IREE_STATUS_FAILED_PRECONDITION;
      }

      // Import that we can fetch from the module state.
      iree_vm_function_call_t call;
      memset(&call, 0, sizeof(call));
      call.function =
          module_state->import_table[function_ordinal & 0x7FFFFFFFu];
      call.argument_registers = src_reg_list;
      call.variadic_segment_size_list = seg_size_list;
      call.result_registers = dst_reg_list;
      IREE_DISPATCH_LOG_CALL(call.function);

      // Call external function.
      iree_status_t call_status = call.function.module->begin_call(
          call.function.module->self, stack, &call, out_result);
      if (!iree_status_is_ok(call_status)) {
        // TODO(benvanik): set execution result to failure/capture stack.
        return call_status;
      }
    });

    DISPATCH_OP(Return, {
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      current_frame->pc = pc;

      // Leave callee by cleaning up the stack.
      iree_vm_stack_function_leave(stack, src_reg_list, &current_frame);

      if (!current_frame || current_frame->depth < entry_frame_depth) {
        // Return from the top-level entry frame - return back to call().
        // TODO(benvanik): clear execution results.
        return IREE_STATUS_OK;
      }

      // Reset dispatch state so we can continue executing in the caller.
      regs = current_frame->registers;
      bytecode_data =
          module->bytecode_data.data +
          module->function_descriptor_table[current_frame->function.ordinal]
              .bytecode_offset;
      pc = current_frame->pc;
    });

    DISPATCH_OP(Fail, {
      uint32_t status_code = VM_DecOperandRegI32("status");
      iree_string_view_t message;
      VM_DecStrAttr("message", &message);
      // TODO(benvanik): attach string and stack.
      if (status_code == 0) {
        // Shouldn't happen; we expect to die here, so there's no way to no-op.
        return IREE_STATUS_INVALID_ARGUMENT;
      }
      return iree_make_status(status_code);
    });

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    DISPATCH_OP(Yield, {
      // TODO(benvanik): yield with execution results.
      return IREE_STATUS_OK;
    });

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    DISPATCH_OP(Trace, {
      iree_string_view_t event_name;
      VM_DecStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      // TODO(benvanik): trace (if enabled).
      iree_vm_bytecode_dispatch_discard_registers(regs, src_reg_list);
    });

    DISPATCH_OP(Print, {
      iree_string_view_t event_name;
      VM_DecStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      // TODO(benvanik): print.
      iree_vm_bytecode_dispatch_discard_registers(regs, src_reg_list);
    });

    DISPATCH_OP(Break, {
      // TODO(benvanik): break unconditionally.
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
      pc = block_pc;
    });

    DISPATCH_OP(CondBreak, {
      int32_t condition = VM_DecOperandRegI32("condition");
      if (condition) {
        // TODO(benvanik): cond break.
      }
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
      pc = block_pc;
    });

    //===------------------------------------------------------------------===//
    // Extension trampolines
    //===------------------------------------------------------------------===//

    DISPATCH_OP(PrefixExtI64, { return IREE_STATUS_UNIMPLEMENTED; });

    DISPATCH_OP(PrefixExtF32, { return IREE_STATUS_UNIMPLEMENTED; });

    DISPATCH_OP(PrefixExtF64, { return IREE_STATUS_UNIMPLEMENTED; });

    // NOLINTNEXTLINE(misc-static-assert)
    DISPATCH_UNHANDLED();
  }
  END_DISPATCH();
}
