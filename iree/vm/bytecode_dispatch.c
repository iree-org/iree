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

// Remaps argument registers from a source list to the 0-N ABI registers.
static void iree_vm_bytecode_dispatch_remap_argument_registers(
    iree_vm_registers_t* src_regs, const iree_vm_register_list_t* src_reg_list,
    iree_vm_registers_t* dst_regs) {
  // Each bank begins left-aligned at 0 and increments per arg of its type.
  int i32_reg_offset = 0;
  int ref_reg_offset = 0;
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      uint16_t dst_reg = ref_reg_offset++;
      memset(&dst_regs->ref[dst_reg & IREE_REF_REGISTER_MASK], 0,
             sizeof(iree_vm_ref_t));
      iree_vm_ref_retain_or_move(
          src_reg & IREE_REF_REGISTER_MOVE_BIT,
          &src_regs->ref[src_reg & IREE_REF_REGISTER_MASK],
          &dst_regs->ref[dst_reg & IREE_REF_REGISTER_MASK]);
    } else {
      uint16_t dst_reg = i32_reg_offset++;
      dst_regs->i32[dst_reg & IREE_I32_REGISTER_MASK] =
          src_regs->i32[src_reg & IREE_I32_REGISTER_MASK];
    }
  }
  dst_regs->ref_register_count = ref_reg_offset;
}

// Remaps registers from source to destination, possibly across frames.
static void iree_vm_bytecode_dispatch_remap_registers(
    iree_vm_registers_t* src_regs, const iree_vm_register_list_t* src_reg_list,
    iree_vm_registers_t* dst_regs,
    const iree_vm_register_list_t* dst_reg_list) {
  VMCHECK(src_reg_list->size == dst_reg_list->size);
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    uint16_t dst_reg = dst_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(
          src_reg & IREE_REF_REGISTER_MOVE_BIT,
          &src_regs->ref[src_reg & IREE_REF_REGISTER_MASK],
          &dst_regs->ref[dst_reg & IREE_REF_REGISTER_MASK]);
    } else {
      dst_regs->i32[dst_reg & IREE_I32_REGISTER_MASK] =
          src_regs->i32[src_reg & IREE_I32_REGISTER_MASK];
    }
  }
}

// Discards ref registers in the list if they are marked move.
static void iree_vm_bytecode_dispatch_discard_registers(
    iree_vm_registers_t* regs, const iree_vm_register_list_t* reg_list) {
  for (int i = 0; i < reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    uint16_t reg = reg_list->registers[i];
    if ((reg & (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) ==
        (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) {
      iree_vm_ref_release(&regs->ref[reg & IREE_REF_REGISTER_MASK]);
    }
  }
}

// Interleaved src-dst register sets.
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

// Remaps registers from a source set to a destination set within the frame.
static void iree_vm_bytecode_dispatch_remap_branch_registers(
    iree_vm_registers_t* regs,
    const iree_vm_register_remap_list_t* remap_list) {
  for (int i = 0; i < remap_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = remap_list->pairs[i].src_reg;
    uint16_t dst_reg = remap_list->pairs[i].dst_reg;
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(src_reg & IREE_REF_REGISTER_MOVE_BIT,
                                 &regs->ref[src_reg & IREE_REF_REGISTER_MASK],
                                 &regs->ref[dst_reg & IREE_REF_REGISTER_MASK]);
    } else {
      regs->i32[dst_reg & IREE_I32_REGISTER_MASK] =
          regs->i32[src_reg & IREE_I32_REGISTER_MASK];
    }
  }
}

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
      IREE_VM_OP_TABLE(DECLARE_DISPATCH_OPC, DECLARE_DISPATCH_RSV)};

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
  case IREE_VM_OP_##op_name:            \
    IREE_DISPATCH_LOG_OPCODE(#op_name); \
    body;                               \
    break;

#endif  // IREE_DISPATCH_MODE_COMPUTED_GOTO

  static const int kRegSize = sizeof(uint16_t);

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

#define OP_R_I32(i) regs->i32[OP_I16(i) & IREE_I32_REGISTER_MASK]
#define OP_R_REF(i) regs->ref[OP_I16(i) & IREE_REF_REGISTER_MASK]
#define OP_R_REF_IS_MOVE(i) (OP_I16(i) & IREE_REF_REGISTER_MOVE_BIT)

  // Primary dispatch state. This is our 'native stack frame' and really
  // just enough to make dereferencing common addresses (like the current
  // offset) faster. You can think of this like CPU state (like PC).
  //
  // The hope is that the compiler decides to keep these in registers (as
  // they are touched for every instruction executed). The frame will change
  // as we call into different functions.
  const iree_vm_function_descriptor_t* entry_function_descriptor =
      &module->function_descriptor_table[entry_frame->function.ordinal];
  iree_vm_stack_frame_t* current_frame = entry_frame;
  const uint8_t* bytecode_data =
      module->bytecode_data.data + entry_function_descriptor->bytecode_offset;
  iree_vm_source_offset_t pc = current_frame->pc;
  iree_vm_registers_t* regs = &current_frame->registers;
  // TODO(benvanik): hide this register initialization logic in the stack enter.
  regs->ref_register_count = entry_function_descriptor->ref_register_count;

  memset(out_result, 0, sizeof(*out_result));

  // NOTE: we should generate this with tblgen, as it has the encoding info.
  // TODO(benvanik): at least generate operand reading/writing and sizes.
  // This could look something like:
  //     OP_GlobalLoadI32_value = OP_GlobalLoadI32_global;
  //     pc += OP_Size_GlobalLoadI32;

  BEGIN_DISPATCH() {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    DISPATCH_OP(GlobalLoadI32, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalLoadI32>,
      //   VM_EncGlobalAttr<"global">,
      //   VM_EncResult<"value">,
      // ];
      int byte_offset = OP_I32(0);
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      const int32_t* global_ptr =
          (const int32_t*)(module_state->rwdata_storage.data + byte_offset);
      OP_R_I32(4) = *global_ptr;
      pc += 4 + kRegSize;
    });
    DISPATCH_OP(GlobalStoreI32, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalStoreI32>,
      //   VM_EncGlobalAttr<"global">,
      //   VM_EncOperand<"value", 0>,
      // ];
      int byte_offset = OP_I32(0);
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int32_t* global_ptr =
          (int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *global_ptr = OP_R_I32(4);
      pc += 4 + kRegSize;
    });

    DISPATCH_OP(GlobalLoadIndirectI32, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalLoadIndirectI32>,
      //   VM_EncOperand<"global", 0>,
      //   VM_EncResult<"value">,
      // ];
      int byte_offset = OP_R_I32(0);
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      const int32_t* global_ptr =
          (const int32_t*)(module_state->rwdata_storage.data + byte_offset);
      OP_R_I32(2) = *global_ptr;
      pc += kRegSize + kRegSize;
    });
    DISPATCH_OP(GlobalStoreIndirectI32, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalStoreIndirectI32>,
      //   VM_EncOperand<"global", 0>,
      //   VM_EncOperand<"value", 1>,
      // ];
      int byte_offset = OP_R_I32(0);
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int32_t* global_ptr =
          (int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *global_ptr = OP_R_I32(2);
      pc += kRegSize + kRegSize;
    });

    DISPATCH_OP(GlobalLoadRef, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalLoadRef>,
      //   VM_EncGlobalAttr<"global">,
      //   VM_EncTypeOf<"value">,
      //   VM_EncResult<"value">,
      // ];
      int global = OP_I32(0);
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int type_id = OP_I32(4);
      type_id = type_id >= module->type_count ? 0 : type_id;
      const iree_vm_type_def_t* type_def = &module->type_table[type_id];
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(OP_R_REF_IS_MOVE(8), global_ref,
                                         type_def->ref_type, &OP_R_REF(8));
      pc += 4 + 4 + kRegSize;
    });
    DISPATCH_OP(GlobalStoreRef, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalStoreRef>,
      //   VM_EncGlobalAttr<"global">,
      //   VM_EncTypeOf<"value">,
      //   VM_EncOperand<"value", 0>,
      // ];
      int global = OP_I32(0);
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int type_id = OP_I32(4);
      type_id = type_id >= module->type_count ? 0 : type_id;
      const iree_vm_type_def_t* type_def = &module->type_table[type_id];
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(OP_R_REF_IS_MOVE(8), &OP_R_REF(8),
                                         type_def->ref_type, global_ref);
      pc += 4 + 4 + kRegSize;
    });

    DISPATCH_OP(GlobalLoadIndirectRef, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalLoadIndirectRef>,
      //   VM_EncOperand<"global", 0>,
      //   VM_EncTypeOf<"value">,
      //   VM_EncResult<"value">,
      // ];
      int global = OP_R_I32(0);
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int type_id = OP_I32(2);
      type_id = type_id >= module->type_count ? 0 : type_id;
      const iree_vm_type_def_t* type_def = &module->type_table[type_id];
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(OP_R_REF_IS_MOVE(6), global_ref,
                                         type_def->ref_type, &OP_R_REF(6));
      pc += kRegSize + 4 + kRegSize;
    });
    DISPATCH_OP(GlobalStoreIndirectRef, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_GlobalStoreIndirectRef>,
      //   VM_EncOperand<"global", 0>,
      //   VM_EncTypeOf<"value">,
      //   VM_EncOperand<"value", 1>,
      // ];
      int global = OP_R_I32(0);
      if (global < 0 || global >= module_state->global_ref_count) {
        return IREE_STATUS_OUT_OF_RANGE;
      }
      int type_id = OP_I32(2);
      type_id = type_id >= module->type_count ? 0 : type_id;
      const iree_vm_type_def_t* type_def = &module->type_table[type_id];
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      iree_vm_ref_retain_or_move_checked(OP_R_REF_IS_MOVE(6), &OP_R_REF(6),
                                         type_def->ref_type, global_ref);
      pc += kRegSize + 4 + kRegSize;
    });

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    DISPATCH_OP(ConstI32, {
      // let encoding = [
      //   VM_EncOpcode<opcode>,
      //   VM_EncIntAttr<"value", type.bitwidth>,
      //   VM_EncResult<"result">,
      // ];
      OP_R_I32(4) = OP_I32(0);
      pc += 4 + kRegSize;
    });

    DISPATCH_OP(ConstI32Zero, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_ConstI32Zero>,
      //   VM_EncResult<"result">,
      // ];
      OP_R_I32(0) = 0;
      pc += kRegSize;
    });

    DISPATCH_OP(ConstRefZero, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_ConstRefZero>,
      //   VM_EncResult<"result">,
      // ];
      iree_vm_ref_release(&OP_R_REF(0));
      pc += kRegSize;
    });

    DISPATCH_OP(ConstRefRodata, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_ConstRefRodata>,
      //   VM_EncRodataAttr<"rodata">,
      //   VM_EncResult<"value">,
      // ];
      int32_t rodata_ordinal = OP_I32(0);
      // TODO(benvanik): allow decompression callbacks to run now (if needed).
      iree_vm_ref_wrap_retain(&module_state->rodata_ref_table[rodata_ordinal],
                              iree_vm_ro_byte_buffer_type_id(), &OP_R_REF(4));
      pc += 4 + kRegSize;
    });

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    DISPATCH_OP(ListAlloc, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListReserve, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListSize, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListResize, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListGetI32, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListSetI32, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListGetRef, { return IREE_STATUS_UNIMPLEMENTED; });
    DISPATCH_OP(ListSetRef, { return IREE_STATUS_UNIMPLEMENTED; });

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    DISPATCH_OP(SelectI32, {
      // let encoding = [
      //   VM_EncOpcode<opcode>,
      //   VM_EncOperand<"condition", 0>,
      //   VM_EncOperand<"true_value", 1>,
      //   VM_EncOperand<"false_value", 2>,
      //   VM_EncResult<"result">,
      // ];
      OP_R_I32(6) = OP_R_I32(0) ? OP_R_I32(2) : OP_R_I32(4);
      pc += kRegSize + kRegSize + kRegSize + kRegSize;
    });

    DISPATCH_OP(SelectRef, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_SelectRef>,
      //   VM_EncOperand<"condition", 0>,
      //   VM_EncTypeOf<"true_value">,
      //   VM_EncOperand<"true_value", 1>,
      //   VM_EncOperand<"false_value", 2>,
      //   VM_EncResult<"result">,
      // ];
      // TODO(benvanik): remove the type_id and use either LHS/RHS (if both are
      // null then output is always null so no need to know the type).
      int type_id = OP_I32(2);
      type_id = type_id >= module->type_count ? 0 : type_id;
      const iree_vm_type_def_t* type_def = &module->type_table[type_id];
      if (OP_R_I32(0)) {
        // Select LHS (+6).
        iree_vm_ref_retain_or_move_checked(OP_R_REF_IS_MOVE(6), &OP_R_REF(6),
                                           type_def->ref_type, &OP_R_REF(10));
        if (OP_R_REF_IS_MOVE(10)) iree_vm_ref_release(&OP_R_REF(8));
      } else {
        // Select RHS (+8).
        iree_vm_ref_retain_or_move_checked(OP_R_REF_IS_MOVE(8), &OP_R_REF(8),
                                           type_def->ref_type, &OP_R_REF(10));
        if (OP_R_REF_IS_MOVE(6)) iree_vm_ref_release(&OP_R_REF(6));
      }
      pc += kRegSize + 4 + kRegSize + kRegSize + kRegSize;
    });

    DISPATCH_OP(SwitchI32, {
      // let encoding = [
      //   VM_EncOpcode<opcode>,
      //   VM_EncOperand<"index", 0>,
      //   VM_EncIntAttr<"default_value", 32>,
      //   VM_EncVariadicOperands<"values">,
      //   VM_EncResult<"result">,
      // ];
      int32_t index = OP_R_I32(0);
      int32_t default_value = OP_I32(2);
      const iree_vm_register_list_t* value_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc + 6];
      pc += kRegSize + 4 + kRegSize + value_reg_list->size * kRegSize;
      int32_t new_value = default_value;
      if (index >= 0 && index < value_reg_list->size) {
        new_value = regs->i32[value_reg_list->registers[index] &
                              IREE_I32_REGISTER_MASK];
      }
      OP_R_I32(0) = new_value;
      pc += kRegSize;
    });

    DISPATCH_OP(SwitchRef, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_SwitchRef>,
      //   VM_EncOperand<"index", 0>,
      //   VM_EncTypeOf<"result">,
      //   VM_EncOperand<"default_value", 1>,
      //   VM_EncVariadicOperands<"values">,
      //   VM_EncResult<"result">,
      // ];
      int32_t index = OP_R_I32(0);
      int32_t type_id = OP_I32(2);
      iree_vm_ref_t* default_value = &OP_R_REF(6);
      int is_move = OP_R_REF_IS_MOVE(6);
      const iree_vm_register_list_t* value_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc + 8];
      // Skip over all operands.
      pc +=
          kRegSize + 4 + kRegSize + kRegSize + value_reg_list->size * kRegSize;
      iree_vm_ref_t* new_value = default_value;
      if (index >= 0 && index < value_reg_list->size) {
        new_value = &regs->ref[value_reg_list->registers[index] &
                               IREE_REF_REGISTER_MASK];
        is_move = value_reg_list->registers[index] & IREE_REF_REGISTER_MOVE_BIT;
      }
      iree_vm_ref_t* result_reg = &OP_R_REF(0);
      pc += kRegSize;
      type_id = type_id >= module->type_count ? 0 : type_id;
      const iree_vm_type_def_t* type_def = &module->type_table[type_id];
      iree_vm_ref_retain_or_move_checked(is_move, new_value, type_def->ref_type,
                                         result_reg);
    });

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

    // let encoding = [
    //   VM_EncOpcode<opcode>,
    //   VM_EncOperand<"operand", 0>,
    //   VM_EncResult<"result">,
    // ];
#define DISPATCH_OP_UNARY_ALU_I32(op_name, type, op) \
  DISPATCH_OP(op_name, {                             \
    OP_R_I32(2) = (int32_t)(op((type)OP_R_I32(0)));  \
    pc += kRegSize + kRegSize;                       \
  });

    // let encoding = [
    //   VM_EncOpcode<opcode>,
    //   VM_EncOperand<"lhs", 0>,
    //   VM_EncOperand<"rhs", 1>,
    //   VM_EncResult<"result">,
    // ];
#define DISPATCH_OP_BINARY_ALU_I32(op_name, type, op)                  \
  DISPATCH_OP(op_name, {                                               \
    OP_R_I32(4) = (int32_t)(((type)OP_R_I32(0))op((type)OP_R_I32(2))); \
    pc += kRegSize + kRegSize + kRegSize;                              \
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

    // let encoding = [
    //   VM_EncOpcode<opcode>,
    //   VM_EncOperand<"operand", 0>,
    //   VM_EncResult<"result">,
    // ];
#define DISPATCH_OP_CAST_I32(op_name, src_type, dst_type) \
  DISPATCH_OP(op_name, {                                  \
    OP_R_I32(2) = (dst_type)((src_type)OP_R_I32(0));      \
    pc += kRegSize + kRegSize;                            \
  });

    DISPATCH_OP_CAST_I32(TruncI8, uint8_t, uint32_t);
    DISPATCH_OP_CAST_I32(TruncI16, uint16_t, uint32_t);
    DISPATCH_OP_CAST_I32(ExtI8I32S, int8_t, int32_t);
    DISPATCH_OP_CAST_I32(ExtI16I32S, int16_t, int32_t);

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

    // let encoding = [
    //   VM_EncOpcode<opcode>,
    //   VM_EncOperand<"operand", 0>,
    //   VM_EncIntAttr<"amount", type.bitwidth>,
    //   VM_EncResult<"result">,
    // ];
#define DISPATCH_OP_SHIFT_I32(op_name, type, op)             \
  DISPATCH_OP(op_name, {                                     \
    OP_R_I32(4) = (int32_t)(((type)OP_R_I32(0))op OP_I8(2)); \
    pc += kRegSize + kRegSize + kRegSize;                    \
  });

    DISPATCH_OP_SHIFT_I32(ShlI32, int32_t, <<);
    DISPATCH_OP_SHIFT_I32(ShrI32S, int32_t, >>);
    DISPATCH_OP_SHIFT_I32(ShrI32U, uint32_t, >>);

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

    // let encoding = [
    //   VM_EncOpcode<opcode>,
    //   VM_EncOperand<"lhs", 0>,
    //   VM_EncOperand<"rhs", 1>,
    //   VM_EncResult<"result">,
    // ];
#define DISPATCH_OP_CMP_I32(op_name, type, op)                        \
  DISPATCH_OP(op_name, {                                              \
    OP_R_I32(4) = (((type)OP_R_I32(0))op((type)OP_R_I32(2))) ? 1 : 0; \
    pc += kRegSize + kRegSize + kRegSize;                             \
  });

    DISPATCH_OP_CMP_I32(CmpEQI32, int32_t, ==);
    DISPATCH_OP_CMP_I32(CmpNEI32, int32_t, !=);
    DISPATCH_OP_CMP_I32(CmpLTI32S, int32_t, <);
    DISPATCH_OP_CMP_I32(CmpLTI32U, uint32_t, <);
    DISPATCH_OP_CMP_I32(CmpLTEI32S, int32_t, <=);
    DISPATCH_OP_CMP_I32(CmpLTEI32U, uint32_t, <=);
    DISPATCH_OP_CMP_I32(CmpGTI32S, int32_t, >);
    DISPATCH_OP_CMP_I32(CmpGTI32U, uint32_t, >);
    DISPATCH_OP_CMP_I32(CmpGTEI32S, int32_t, >=);
    DISPATCH_OP_CMP_I32(CmpGTEI32U, uint32_t, >=);

    DISPATCH_OP(CmpEQRef, {
      // let encoding = [
      //   VM_EncOpcode<opcode>,
      //   VM_EncOperand<"lhs", 0>,
      //   VM_EncOperand<"rhs", 1>,
      //   VM_EncResult<"result">,
      // ];
      // TODO(benvanik): move refs.
      OP_R_I32(4) = iree_vm_ref_equal(&OP_R_REF(0), &OP_R_REF(2));
      if (OP_R_REF_IS_MOVE(0)) iree_vm_ref_release(&OP_R_REF(0));
      if (OP_R_REF_IS_MOVE(2)) iree_vm_ref_release(&OP_R_REF(2));
      pc += kRegSize + kRegSize + kRegSize;
    });
    DISPATCH_OP(CmpNERef, {
      // let encoding = [
      //   VM_EncOpcode<opcode>,
      //   VM_EncOperand<"lhs", 0>,
      //   VM_EncOperand<"rhs", 1>,
      //   VM_EncResult<"result">,
      // ];
      OP_R_I32(4) = !iree_vm_ref_equal(&OP_R_REF(0), &OP_R_REF(2));
      if (OP_R_REF_IS_MOVE(0)) iree_vm_ref_release(&OP_R_REF(0));
      if (OP_R_REF_IS_MOVE(2)) iree_vm_ref_release(&OP_R_REF(2));
      pc += kRegSize + kRegSize + kRegSize;
    });
    DISPATCH_OP(CmpNZRef, {
      // let encoding = [
      //   VM_EncOpcode<opcode>,
      //   VM_EncOperand<"operand", 0>,
      //   VM_EncResult<"result">,
      // ];
      OP_R_I32(2) = OP_R_REF(0).ptr != NULL;
      if (OP_R_REF_IS_MOVE(0)) iree_vm_ref_release(&OP_R_REF(0));
      pc += kRegSize + kRegSize;
    });

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    DISPATCH_OP(Branch, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Branch>,
      //   VM_EncBranch<"dest", "operands">,
      // ];

      int32_t block_pc = OP_I32(0);
      const iree_vm_register_remap_list_t* remap_list =
          (const iree_vm_register_remap_list_t*)&bytecode_data[pc + 4];
      pc += 4 + kRegSize + remap_list->size * 2 * kRegSize;
      pc = block_pc;
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
    });

    DISPATCH_OP(CondBranch, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_CondBranch>,
      //   VM_EncOperand<"condition", 0>,
      //   VM_EncBranch<"getTrueDest", "getTrueOperands">,
      //   VM_EncBranch<"getFalseDest", "getFalseOperands">,
      // ];

      int32_t cond_value = OP_R_I32(0);
      int32_t true_block_pc = OP_I32(2);
      const iree_vm_register_remap_list_t* true_remap_list =
          (const iree_vm_register_remap_list_t*)&bytecode_data[pc + kRegSize +
                                                               4];
      pc += kRegSize + 4 + kRegSize + true_remap_list->size * 2 * kRegSize;
      int32_t false_block_pc = OP_I32(0);
      const iree_vm_register_remap_list_t* false_remap_list =
          (const iree_vm_register_remap_list_t*)&bytecode_data[pc + 4];
      pc += 4 + kRegSize + false_remap_list->size * 2 * kRegSize;

      if (cond_value) {
        pc = true_block_pc;
        iree_vm_bytecode_dispatch_remap_branch_registers(regs, true_remap_list);
      } else {
        pc = false_block_pc;
        iree_vm_bytecode_dispatch_remap_branch_registers(regs,
                                                         false_remap_list);
      }
    });

    DISPATCH_OP(Call, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Call>,
      //   VM_EncFuncAttr<"callee">,
      //   VM_EncVariadicOperands<"operands">,
      //   VM_EncVariadicResults<"results">,
      // ];

      // Get argument and result register lists and flush the caller frame.
      int32_t function_ordinal = OP_I32(0);
      const iree_vm_register_list_t* src_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc + 4];
      pc += 4 + kRegSize + src_reg_list->size * kRegSize;
      const iree_vm_register_list_t* dst_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      current_frame->return_registers = dst_reg_list;
      pc += kRegSize + dst_reg_list->size * kRegSize;
      current_frame->pc = pc;

      // NOTE: we assume validation has ensured these functions exist.
      // TODO(benvanik): something more clever than just a high bit?
      iree_vm_function_t target_function;
      int is_import = (function_ordinal & 0x80000000u) != 0;
      if (is_import) {
        // Import that we can fetch from the module state.
        target_function =
            module_state->import_table[function_ordinal & 0x7FFFFFFFu];
      } else {
        // Internal to the current module.
        target_function.module = &module->interface;
        target_function.linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
        target_function.ordinal = function_ordinal;
      }

      IREE_DISPATCH_LOG_CALL(target_function);

      // Remap registers from caller to callee.
      iree_vm_stack_frame_t* callee_frame = NULL;
      iree_status_t enter_status =
          iree_vm_stack_function_enter(stack, target_function, &callee_frame);
      if (!iree_status_is_ok(enter_status)) {
        // TODO(benvanik): set execution result to stack overflow.
        return enter_status;
      }
      iree_vm_bytecode_dispatch_remap_argument_registers(
          &current_frame->registers, src_reg_list, &callee_frame->registers);

      if (is_import) {
        // Call external function.
        iree_status_t call_status = target_function.module->execute(
            target_function.module->self, stack, callee_frame, out_result);
        if (!iree_status_is_ok(call_status)) {
          // TODO(benvanik): set execution result to failure/capture stack.
          return call_status;
        }
        if (callee_frame->return_registers) {
          iree_vm_bytecode_dispatch_remap_registers(
              &callee_frame->registers, callee_frame->return_registers,
              &current_frame->registers, current_frame->return_registers);
        }
        iree_vm_stack_function_leave(stack);
      } else {
        // Switch execution to the target function and continue running in the
        // bytecode dispatcher.
        const iree_vm_function_descriptor_t* function_descriptor =
            &module->function_descriptor_table[callee_frame->function.ordinal];
        current_frame = callee_frame;
        bytecode_data =
            module->bytecode_data.data + function_descriptor->bytecode_offset;
        regs = &callee_frame->registers;
        // TODO(benvanik): hide this in the stack.
        memset(
            &regs->ref[regs->ref_register_count], 0,
            sizeof(iree_vm_ref_t) * (function_descriptor->ref_register_count -
                                     regs->ref_register_count));
        regs->ref_register_count = function_descriptor->ref_register_count;
        pc = callee_frame->pc;
      }
    });

    DISPATCH_OP(CallVariadic, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_CallVariadic>,
      //   VM_EncFuncAttr<"callee">,
      //   VM_EncIntArrayAttr<"segment_sizes", 16>,
      //   VM_EncVariadicOperands<"operands">,
      //   VM_EncVariadicResults<"results">,
      // ];

      // TODO(benvanik): dedupe with above or merge and always have the seg size
      // list be present (but empty) for non-variadic calls.

      // Get argument and result register lists and flush the caller frame.
      int32_t function_ordinal = OP_I32(0);
      pc += 4;
      const iree_vm_register_list_t* seg_size_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      pc += kRegSize + seg_size_list->size * kRegSize;
      const iree_vm_register_list_t* src_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      pc += kRegSize + src_reg_list->size * kRegSize;
      const iree_vm_register_list_t* dst_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      current_frame->return_registers = dst_reg_list;
      pc += kRegSize + dst_reg_list->size * kRegSize;
      current_frame->pc = pc;

      // NOTE: we assume validation has ensured these functions exist.
      // TODO(benvanik): something more clever than just a high bit?
      iree_vm_function_t target_function;
      int is_import = (function_ordinal & 0x80000000u) != 0;
      if (!is_import) {
        // Variadic calls are currently only supported for import functions.
        return IREE_STATUS_FAILED_PRECONDITION;
      }

      // Import that we can fetch from the module state.
      target_function =
          module_state->import_table[function_ordinal & 0x7FFFFFFFu];

      IREE_DISPATCH_LOG_CALL(target_function);

      // Remap registers from caller to callee.
      iree_vm_stack_frame_t* callee_frame = NULL;
      iree_status_t enter_status =
          iree_vm_stack_function_enter(stack, target_function, &callee_frame);
      if (!iree_status_is_ok(enter_status)) {
        // TODO(benvanik): set execution result to stack overflow.
        return enter_status;
      }
      iree_vm_bytecode_dispatch_remap_argument_registers(
          &current_frame->registers, src_reg_list, &callee_frame->registers);

      // TODO(benvanik): rename return_registers.
      callee_frame->return_registers = seg_size_list;

      // Call external function.
      iree_status_t call_status = target_function.module->execute(
          target_function.module->self, stack, callee_frame, out_result);
      if (!iree_status_is_ok(call_status)) {
        // TODO(benvanik): set execution result to failure/capture stack.
        return call_status;
      }
      if (callee_frame->return_registers) {
        iree_vm_bytecode_dispatch_remap_registers(
            &callee_frame->registers, callee_frame->return_registers,
            &current_frame->registers, current_frame->return_registers);
      }
      iree_vm_stack_function_leave(stack);
    });

    DISPATCH_OP(Return, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Return>,
      //   VM_EncVariadicOperands<"operands">,
      // ];

      // Remap registers from callee to caller.
      const iree_vm_register_list_t* src_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      current_frame->pc = pc + kRegSize + src_reg_list->size * kRegSize;

      if (current_frame == entry_frame) {
        // Return from the top-level entry frame - return back to execute().
        // TODO(benvanik): clear execution results.
        current_frame->return_registers = src_reg_list;
        return IREE_STATUS_OK;
      }

      // Copy results back to the caller registers.
      // The caller pc should be pointing at the head of the return
      // registers list.
      iree_vm_stack_frame_t* caller_frame = iree_vm_stack_parent_frame(stack);
      VMCHECK(caller_frame);
      iree_vm_bytecode_dispatch_remap_registers(
          &current_frame->registers, src_reg_list, &caller_frame->registers,
          caller_frame->return_registers);

      // Leave callee by cleaning up the stack.
      iree_vm_stack_function_leave(stack);

      // Reset dispatch state so we can continue executing in the caller.
      current_frame = caller_frame;
      bytecode_data =
          module->bytecode_data.data +
          module->function_descriptor_table[caller_frame->function.ordinal]
              .bytecode_offset;
      regs = &caller_frame->registers;
      pc = caller_frame->pc;
    });

    DISPATCH_OP(Fail, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Fail>,
      //   VM_EncOperand<"status", 0>,
      //   VM_EncStrAttr<"message">,
      // ];
      uint32_t status_code = OP_I8(0);
      iree_string_view_t str;
      str.size = OP_I16(1);
      str.data = (const char*)&bytecode_data[pc + 3];
      pc += 1 + 2 + str.size;
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
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Yield>,
      // ];
      // TODO(benvanik): yield with execution results.
      return IREE_STATUS_OK;
    });

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    DISPATCH_OP(Trace, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Trace>,
      //   VM_EncStrAttr<"event_name">,
      //   VM_EncVariadicOperands<"operands">,
      // ];
      iree_string_view_t str;
      str.size = OP_I16(0);
      str.data = (const char*)&bytecode_data[pc + 2];
      pc += 2 + str.size;
      const iree_vm_register_list_t* src_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      pc += kRegSize + src_reg_list->size * kRegSize;
      // TODO(benvanik): trace (if enabled).
      iree_vm_bytecode_dispatch_discard_registers(regs, src_reg_list);
    });

    DISPATCH_OP(Print, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Print>,
      //   VM_EncStrAttr<"message">,
      //   VM_EncVariadicOperands<"operands">,
      // ];
      iree_string_view_t str;
      str.size = OP_I16(0);
      str.data = (const char*)&bytecode_data[pc + 2];
      pc += 2 + str.size;
      const iree_vm_register_list_t* src_reg_list =
          (const iree_vm_register_list_t*)&bytecode_data[pc];
      pc += kRegSize + src_reg_list->size * kRegSize;
      // TODO(benvanik): print.
      iree_vm_bytecode_dispatch_discard_registers(regs, src_reg_list);
    });

    DISPATCH_OP(Break, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_Break>,
      //   VM_EncBranch<"dest", "operands">,
      // ];
      // TODO(benvanik): break unconditionally.
      int32_t block_pc = OP_I32(0);
      const iree_vm_register_remap_list_t* remap_list =
          (const iree_vm_register_remap_list_t*)&bytecode_data[pc + 4];
      pc += 4 + kRegSize + remap_list->size * 2 * kRegSize;
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
      pc = block_pc;
    });

    DISPATCH_OP(CondBreak, {
      // let encoding = [
      //   VM_EncOpcode<VM_OPC_CondBreak>,
      //   VM_EncBranch<"dest", "operands">,
      // ];
      int32_t cond_value = OP_R_I32(0);
      if (cond_value) {
        // TODO(benvanik): cond break.
      }
      int32_t block_pc = OP_I32(2);
      const iree_vm_register_remap_list_t* remap_list =
          (const iree_vm_register_remap_list_t*)&bytecode_data[pc + kRegSize +
                                                               4];
      pc += kRegSize + 4 + kRegSize + remap_list->size * 2 * kRegSize;
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
      pc = block_pc;
    });

    // NOLINTNEXTLINE(misc-static-assert)
    DISPATCH_UNHANDLED();
  }
  END_DISPATCH();
}
