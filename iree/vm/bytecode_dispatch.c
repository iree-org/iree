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

#include <string.h>

#include "iree/vm/bytecode_dispatch_util.h"
#include "iree/vm/list.h"

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

iree_status_t iree_vm_bytecode_dispatch(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, iree_vm_stack_t* stack,
    iree_vm_stack_frame_t* entry_frame,
    iree_vm_execution_result_t* out_result) {
  // When required emit the dispatch tables here referencing the labels we are
  // defining below.
  DEFINE_DISPATCH_TABLES();

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

  BEGIN_DISPATCH_CORE() {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, GlobalLoadI32, {
      int32_t byte_offset = VM_DecGlobalAttr("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int32_t* value = VM_DecResultRegI32("value");
      const int32_t* global_ptr =
          (const int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *value = *global_ptr;
    });

    DISPATCH_OP(CORE, GlobalStoreI32, {
      int32_t byte_offset = VM_DecGlobalAttr("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int32_t value = VM_DecOperandRegI32("value");
      int32_t* global_ptr =
          (int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *global_ptr = value;
    });

    DISPATCH_OP(CORE, GlobalLoadIndirectI32, {
      int32_t byte_offset = VM_DecOperandRegI32("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int32_t* value = VM_DecResultRegI32("value");
      const int32_t* global_ptr =
          (const int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *value = *global_ptr;
    });

    DISPATCH_OP(CORE, GlobalStoreIndirectI32, {
      int32_t byte_offset = VM_DecOperandRegI32("global");
      if (byte_offset < 0 ||
          byte_offset >= module_state->rwdata_storage.data_length) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int32_t value = VM_DecOperandRegI32("value");
      int32_t* global_ptr =
          (int32_t*)(module_state->rwdata_storage.data + byte_offset);
      *global_ptr = value;
    });

    DISPATCH_OP(CORE, GlobalLoadRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global ref ordinal out of range: %d (table=%zu)", global,
            module_state->global_ref_count);
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          result_is_move, global_ref, type_def->ref_type, result));
    });

    DISPATCH_OP(CORE, GlobalStoreRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global ref ordinal out of range: %d (table=%zu)", global,
            module_state->global_ref_count);
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool value_is_move;
      iree_vm_ref_t* value = VM_DecOperandRegRef("value", &value_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          value_is_move, value, type_def->ref_type, global_ref));
    });

    DISPATCH_OP(CORE, GlobalLoadIndirectRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global ref ordinal out of range: %d (table=%zu)", global,
            module_state->global_ref_count);
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          result_is_move, global_ref, type_def->ref_type, result));
    });

    DISPATCH_OP(CORE, GlobalStoreIndirectRef, {
      int32_t global = VM_DecGlobalAttr("global");
      if (global < 0 || global >= module_state->global_ref_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global ref ordinal out of range: %d (table=%zu)", global,
            module_state->global_ref_count);
      }
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool value_is_move;
      iree_vm_ref_t* value = VM_DecOperandRegRef("value", &value_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          value_is_move, value, type_def->ref_type, global_ref));
    });

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, ConstI32, {
      int32_t value = VM_DecIntAttr32("value");
      int32_t* result = VM_DecResultRegI32("result");
      *result = value;
    });

    DISPATCH_OP(CORE, ConstI32Zero, {
      int32_t* result = VM_DecResultRegI32("result");
      *result = 0;
    });

    DISPATCH_OP(CORE, ConstRefZero, {
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_ref_release(result);
    });

    DISPATCH_OP(CORE, ConstRefRodata, {
      int32_t rodata_ordinal = VM_DecRodataAttr("rodata");
      if (rodata_ordinal < 0 ||
          rodata_ordinal >= module_state->rodata_ref_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "rodata ref ordinal out of range: %d (table=%zu)", rodata_ordinal,
            module_state->rodata_ref_count);
      }
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_retain(
          &module_state->rodata_ref_table[rodata_ordinal],
          iree_vm_ro_byte_buffer_type_id(), result));
    });

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, ListAlloc, {
      const iree_vm_type_def_t* element_type_def = VM_DecTypeOf("element_type");
      iree_host_size_t initial_capacity =
          VM_DecOperandRegI32("initial_capacity");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_list_t* list = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_list_create(
          element_type_def, initial_capacity, module_state->allocator, &list));
      IREE_RETURN_IF_ERROR(
          iree_vm_ref_wrap_assign(list, iree_vm_list_type_id(), result));
    });

    DISPATCH_OP(CORE, ListReserve, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t minimum_capacity = VM_DecOperandRegI32("minimum_capacity");
      IREE_RETURN_IF_ERROR(iree_vm_list_reserve(list, minimum_capacity));
    });

    DISPATCH_OP(CORE, ListSize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t* result = VM_DecResultRegI32("result");
      *result = (int32_t)iree_vm_list_size(list);
    });

    DISPATCH_OP(CORE, ListResize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t new_size = VM_DecOperandRegI32("new_size");
      IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, new_size));
    });

    DISPATCH_OP(CORE, ListGetI32, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t index = VM_DecOperandRegI32("index");
      int32_t* result = VM_DecResultRegI32("result");
      iree_vm_value_t value;
      IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
          list, index, IREE_VM_VALUE_TYPE_I32, &value));
      *result = value.i32;
    });

    DISPATCH_OP(CORE, ListSetI32, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (!list) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t index = VM_DecOperandRegI32("index");
      int32_t raw_value = VM_DecOperandRegI32("raw_value");
      iree_vm_value_t value = iree_vm_value_make_i32(raw_value);
      IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
    });

    DISPATCH_OP(CORE, ListGetRef, {
      // bool list_is_move;
      // iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      // iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      // if (!list) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      // int32_t index = VM_DecOperandRegI32("index");
      // iree_vm_ref_t* result = VM_DecResultRegRef("result");
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "vm.list.get.ref not implemented");
    });

    DISPATCH_OP(CORE, ListSetRef, {
      // bool list_is_move;
      // iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      // iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      // if (!list) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      // int32_t index = VM_DecOperandRegI32("index");
      // bool operand_is_move = VM_DecOperandRegRefIsMove("value");
      // iree_vm_ref_t* operand = VM_DecOperandRegRef("value");
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "vm.list.set.ref not implemented");
    });

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, SelectI32, {
      int32_t condition = VM_DecOperandRegI32("condition");
      int32_t true_value = VM_DecOperandRegI32("true_value");
      int32_t false_value = VM_DecOperandRegI32("false_value");
      int32_t* result = VM_DecResultRegI32("result");
      *result = condition ? true_value : false_value;
    });

    DISPATCH_OP(CORE, SelectRef, {
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
        IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
            true_value_is_move, true_value, type_def->ref_type, result));
        if (false_value_is_move) iree_vm_ref_release(false_value);
      } else {
        // Select RHS.
        IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
            false_value_is_move, false_value, type_def->ref_type, result));
        if (true_value_is_move) iree_vm_ref_release(true_value);
      }
    });

    DISPATCH_OP(CORE, SwitchI32, {
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

    DISPATCH_OP(CORE, SwitchRef, {
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
        IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
            is_move, new_value, type_def->ref_type, result));
      } else {
        IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
            default_is_move, default_value, type_def->ref_type, result));
      }
    });

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CORE_UNARY_ALU_I32(op_name, type, op) \
  DISPATCH_OP(CORE, op_name, {                            \
    int32_t operand = VM_DecOperandRegI32("operand");     \
    int32_t* result = VM_DecResultRegI32("result");       \
    *result = (int32_t)(op((type)operand));               \
  });

#define DISPATCH_OP_CORE_BINARY_ALU_I32(op_name, type, op) \
  DISPATCH_OP(CORE, op_name, {                             \
    int32_t lhs = VM_DecOperandRegI32("lhs");              \
    int32_t rhs = VM_DecOperandRegI32("rhs");              \
    int32_t* result = VM_DecResultRegI32("result");        \
    *result = (int32_t)(((type)lhs)op((type)rhs));         \
  });

    DISPATCH_OP_CORE_BINARY_ALU_I32(AddI32, int32_t, +);
    DISPATCH_OP_CORE_BINARY_ALU_I32(SubI32, int32_t, -);
    DISPATCH_OP_CORE_BINARY_ALU_I32(MulI32, int32_t, *);
    DISPATCH_OP_CORE_BINARY_ALU_I32(DivI32S, int32_t, /);
    DISPATCH_OP_CORE_BINARY_ALU_I32(DivI32U, uint32_t, /);
    DISPATCH_OP_CORE_BINARY_ALU_I32(RemI32S, int32_t, %);
    DISPATCH_OP_CORE_BINARY_ALU_I32(RemI32U, uint32_t, %);
    DISPATCH_OP_CORE_UNARY_ALU_I32(NotI32, uint32_t, ~);
    DISPATCH_OP_CORE_BINARY_ALU_I32(AndI32, uint32_t, &);
    DISPATCH_OP_CORE_BINARY_ALU_I32(OrI32, uint32_t, |);
    DISPATCH_OP_CORE_BINARY_ALU_I32(XorI32, uint32_t, ^);

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CORE_CAST_I32(op_name, src_type, dst_type) \
  DISPATCH_OP(CORE, op_name, {                                 \
    int32_t operand = VM_DecOperandRegI32("operand");          \
    int32_t* result = VM_DecResultRegI32("result");            \
    *result = (dst_type)((src_type)operand);                   \
  });

    DISPATCH_OP_CORE_CAST_I32(TruncI32I8, uint32_t, uint8_t);
    DISPATCH_OP_CORE_CAST_I32(TruncI32I16, uint32_t, uint16_t);
    DISPATCH_OP_CORE_CAST_I32(ExtI8I32S, int8_t, int32_t);
    DISPATCH_OP_CORE_CAST_I32(ExtI8I32U, uint8_t, uint32_t);
    DISPATCH_OP_CORE_CAST_I32(ExtI16I32S, int16_t, int32_t);
    DISPATCH_OP_CORE_CAST_I32(ExtI16I32U, uint16_t, uint32_t);

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CORE_SHIFT_I32(op_name, type, op) \
  DISPATCH_OP(CORE, op_name, {                        \
    int32_t operand = VM_DecOperandRegI32("operand"); \
    int8_t amount = VM_DecConstI8("amount");          \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = (int32_t)(((type)operand)op amount);    \
  });

    DISPATCH_OP_CORE_SHIFT_I32(ShlI32, int32_t, <<);
    DISPATCH_OP_CORE_SHIFT_I32(ShrI32S, int32_t, >>);
    DISPATCH_OP_CORE_SHIFT_I32(ShrI32U, uint32_t, >>);

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CORE_CMP_I32(op_name, type, op) \
  DISPATCH_OP(CORE, op_name, {                      \
    int32_t lhs = VM_DecOperandRegI32("lhs");       \
    int32_t rhs = VM_DecOperandRegI32("rhs");       \
    int32_t* result = VM_DecResultRegI32("result"); \
    *result = (((type)lhs)op((type)rhs)) ? 1 : 0;   \
  });

    DISPATCH_OP_CORE_CMP_I32(CmpEQI32, int32_t, ==);
    DISPATCH_OP_CORE_CMP_I32(CmpNEI32, int32_t, !=);
    DISPATCH_OP_CORE_CMP_I32(CmpLTI32S, int32_t, <);
    DISPATCH_OP_CORE_CMP_I32(CmpLTI32U, uint32_t, <);
    DISPATCH_OP(CORE, CmpNZI32, {
      int32_t operand = VM_DecOperandRegI32("operand");
      int32_t* result = VM_DecResultRegI32("result");
      *result = (operand != 0) ? 1 : 0;
    });

    DISPATCH_OP(CORE, CmpEQRef, {
      bool lhs_is_move;
      iree_vm_ref_t* lhs = VM_DecOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      iree_vm_ref_t* rhs = VM_DecOperandRegRef("rhs", &rhs_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = iree_vm_ref_equal(lhs, rhs);
      if (lhs_is_move) iree_vm_ref_release(lhs);
      if (rhs_is_move) iree_vm_ref_release(rhs);
    });
    DISPATCH_OP(CORE, CmpNERef, {
      bool lhs_is_move;
      iree_vm_ref_t* lhs = VM_DecOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      iree_vm_ref_t* rhs = VM_DecOperandRegRef("rhs", &rhs_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = !iree_vm_ref_equal(lhs, rhs);
      if (lhs_is_move) iree_vm_ref_release(lhs);
      if (rhs_is_move) iree_vm_ref_release(rhs);
    });
    DISPATCH_OP(CORE, CmpNZRef, {
      bool operand_is_move;
      iree_vm_ref_t* operand = VM_DecOperandRegRef("operand", &operand_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = operand->ptr != NULL ? 1 : 0;
      if (operand_is_move) iree_vm_ref_release(operand);
    });

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, Branch, {
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      pc = block_pc;
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
    });

    DISPATCH_OP(CORE, CondBranch, {
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

    DISPATCH_OP(CORE, Call, {
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
          return iree_status_annotate(
              call_status, iree_make_cstring_view("while calling import"));
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
          return iree_status_annotate(
              enter_status,
              iree_make_cstring_view("while calling internal function"));
        }
        regs = current_frame->registers;
        bytecode_data =
            module->bytecode_data.data + target_descriptor->bytecode_offset;
        pc = current_frame->pc;
      }
    });

    DISPATCH_OP(CORE, CallVariadic, {
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
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "variadic calls only supported for internal callees");
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
        return iree_status_annotate(
            call_status,
            iree_make_cstring_view("while calling variadic import"));
      }
    });

    DISPATCH_OP(CORE, Return, {
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      current_frame->pc = pc;

      // Leave callee by cleaning up the stack.
      IREE_RETURN_IF_ERROR(
          iree_vm_stack_function_leave(stack, src_reg_list, &current_frame));

      if (!current_frame || current_frame->depth < entry_frame_depth) {
        // Return from the top-level entry frame - return back to call().
        // TODO(benvanik): clear execution results.
        return iree_ok_status();
      }

      // Reset dispatch state so we can continue executing in the caller.
      regs = current_frame->registers;
      bytecode_data =
          module->bytecode_data.data +
          module->function_descriptor_table[current_frame->function.ordinal]
              .bytecode_offset;
      pc = current_frame->pc;
    });

    DISPATCH_OP(CORE, Fail, {
      uint32_t status_code = VM_DecOperandRegI32("status");
      iree_string_view_t message;
      VM_DecStrAttr("message", &message);
      if (status_code != 0) {
        // TODO(benvanik): capture source information.
        return iree_status_allocate(status_code, "<vm>", 0, message);
      }
    });

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, Yield, {
      // TODO(benvanik): yield with execution results.
      return iree_ok_status();
    });

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, Trace, {
      iree_string_view_t event_name;
      VM_DecStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      // TODO(benvanik): trace (if enabled).
      iree_vm_bytecode_dispatch_discard_registers(regs, src_reg_list);
    });

    DISPATCH_OP(CORE, Print, {
      iree_string_view_t event_name;
      VM_DecStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      // TODO(benvanik): print.
      iree_vm_bytecode_dispatch_discard_registers(regs, src_reg_list);
    });

    DISPATCH_OP(CORE, Break, {
      // TODO(benvanik): break unconditionally.
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      iree_vm_bytecode_dispatch_remap_branch_registers(regs, remap_list);
      pc = block_pc;
    });

    DISPATCH_OP(CORE, CondBreak, {
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

    BEGIN_DISPATCH_PREFIX(PrefixExtI64, EXT_I64) {
#if IREE_VM_EXT_I64_ENABLE
      //===----------------------------------------------------------------===//
      // ExtI64: Globals
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_I64, GlobalLoadI64, {
        int32_t byte_offset = VM_DecGlobalAttr("global");
        if (byte_offset < 0 ||
            byte_offset >= module_state->rwdata_storage.data_length) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
              module_state->rwdata_storage.data_length);
        }
        int64_t* value = VM_DecResultRegI64("value");
        const int64_t* global_ptr =
            (const int64_t*)(module_state->rwdata_storage.data + byte_offset);
        *value = *global_ptr;
      });

      DISPATCH_OP(EXT_I64, GlobalStoreI64, {
        int32_t byte_offset = VM_DecGlobalAttr("global");
        if (byte_offset < 0 ||
            byte_offset >= module_state->rwdata_storage.data_length) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
              module_state->rwdata_storage.data_length);
        }
        int64_t value = VM_DecOperandRegI64("value");
        int64_t* global_ptr =
            (int64_t*)(module_state->rwdata_storage.data + byte_offset);
        *global_ptr = value;
      });

      DISPATCH_OP(EXT_I64, GlobalLoadIndirectI64, {
        int32_t byte_offset = VM_DecOperandRegI32("global");
        if (byte_offset < 0 ||
            byte_offset >= module_state->rwdata_storage.data_length) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
              module_state->rwdata_storage.data_length);
        }
        int64_t* value = VM_DecResultRegI64("value");
        const int64_t* global_ptr =
            (const int64_t*)(module_state->rwdata_storage.data + byte_offset);
        *value = *global_ptr;
      });

      DISPATCH_OP(EXT_I64, GlobalStoreIndirectI64, {
        int32_t byte_offset = VM_DecOperandRegI32("global");
        if (byte_offset < 0 ||
            byte_offset >= module_state->rwdata_storage.data_length) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
              module_state->rwdata_storage.data_length);
        }
        int64_t value = VM_DecOperandRegI64("value");
        int64_t* global_ptr =
            (int64_t*)(module_state->rwdata_storage.data + byte_offset);
        *global_ptr = value;
      });

      //===----------------------------------------------------------------===//
      // ExtI64: Constants
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_I64, ConstI64, {
        int64_t value = VM_DecIntAttr64("value");
        int64_t* result = VM_DecResultRegI64("result");
        *result = value;
      });

      DISPATCH_OP(EXT_I64, ConstI64Zero, {
        int64_t* result = VM_DecResultRegI64("result");
        *result = 0;
      });

      //===----------------------------------------------------------------===//
      // ExtI64: Lists
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_I64, ListGetI64, {
        bool list_is_move;
        iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
        iree_vm_list_t* list = iree_vm_list_deref(list_ref);
        if (!list) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        int32_t index = VM_DecOperandRegI32("index");
        int64_t* result = VM_DecResultRegI64("result");
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            list, index, IREE_VM_VALUE_TYPE_I64, &value));
        *result = value.i32;
      });

      DISPATCH_OP(EXT_I64, ListSetI64, {
        bool list_is_move;
        iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
        iree_vm_list_t* list = iree_vm_list_deref(list_ref);
        if (!list) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        int32_t index = VM_DecOperandRegI32("index");
        int64_t raw_value = VM_DecOperandRegI64("value");
        iree_vm_value_t value = iree_vm_value_make_i64(raw_value);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
      });

      //===----------------------------------------------------------------===//
      // ExtI64: Conditional assignment
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_I64, SelectI64, {
        int32_t condition = VM_DecOperandRegI32("condition");
        int64_t true_value = VM_DecOperandRegI64("true_value");
        int64_t false_value = VM_DecOperandRegI64("false_value");
        int64_t* result = VM_DecResultRegI64("result");
        *result = condition ? true_value : false_value;
      });

      DISPATCH_OP(EXT_I64, SwitchI64, {
        int32_t index = VM_DecOperandRegI32("index");
        int64_t default_value = VM_DecIntAttr64("default_value");
        const iree_vm_register_list_t* value_reg_list =
            VM_DecVariadicOperands("values");
        int64_t* result = VM_DecResultRegI64("result");
        if (index >= 0 && index < value_reg_list->size) {
          *result =
              regs.i32[value_reg_list->registers[index] & (regs.i32_mask & ~1)];
        } else {
          *result = default_value;
        }
      });

      //===----------------------------------------------------------------===//
      // ExtI64: Native integer arithmetic
      //===----------------------------------------------------------------===//

#define DISPATCH_OP_EXT_I64_UNARY_ALU_I64(op_name, type, op) \
  DISPATCH_OP(EXT_I64, op_name, {                            \
    int64_t operand = VM_DecOperandRegI64("operand");        \
    int64_t* result = VM_DecResultRegI64("result");          \
    *result = (int64_t)(op((type)operand));                  \
  });

#define DISPATCH_OP_EXT_I64_BINARY_ALU_I64(op_name, type, op) \
  DISPATCH_OP(EXT_I64, op_name, {                             \
    int64_t lhs = VM_DecOperandRegI64("lhs");                 \
    int64_t rhs = VM_DecOperandRegI64("rhs");                 \
    int64_t* result = VM_DecResultRegI64("result");           \
    *result = (int64_t)(((type)lhs)op((type)rhs));            \
  });

      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(AddI64, int64_t, +);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(SubI64, int64_t, -);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(MulI64, int64_t, *);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(DivI64S, int64_t, /);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(DivI64U, uint64_t, /);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(RemI64S, int64_t, %);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(RemI64U, uint64_t, %);
      DISPATCH_OP_EXT_I64_UNARY_ALU_I64(NotI64, uint64_t, ~);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(AndI64, uint64_t, &);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(OrI64, uint64_t, |);
      DISPATCH_OP_EXT_I64_BINARY_ALU_I64(XorI64, uint64_t, ^);

      //===----------------------------------------------------------------===//
      // ExtI64: Casting and type conversion/emulation
      //===----------------------------------------------------------------===//

#define DISPATCH_OP_EXT_I64_CAST_I64(op_name, src_type, dst_type) \
  DISPATCH_OP(EXT_I64, op_name, {                                 \
    int64_t operand = VM_DecOperandRegI64("operand");             \
    int64_t* result = VM_DecResultRegI64("result");               \
    *result = (dst_type)((src_type)operand);                      \
  });

      DISPATCH_OP_EXT_I64_CAST_I64(TruncI64I32, uint64_t, uint32_t);
      DISPATCH_OP_EXT_I64_CAST_I64(ExtI32I64S, int32_t, int64_t);
      DISPATCH_OP_EXT_I64_CAST_I64(ExtI32I64U, uint32_t, uint64_t);

      //===----------------------------------------------------------------===//
      // ExtI64: Native bitwise shifts and rotates
      //===----------------------------------------------------------------===//

#define DISPATCH_OP_EXT_I64_SHIFT_I64(op_name, type, op) \
  DISPATCH_OP(EXT_I64, op_name, {                        \
    int64_t operand = VM_DecOperandRegI64("operand");    \
    int8_t amount = VM_DecConstI8("amount");             \
    int64_t* result = VM_DecResultRegI64("result");      \
    *result = (int64_t)(((type)operand)op amount);       \
  });

      DISPATCH_OP_EXT_I64_SHIFT_I64(ShlI64, int64_t, <<);
      DISPATCH_OP_EXT_I64_SHIFT_I64(ShrI64S, int64_t, >>);
      DISPATCH_OP_EXT_I64_SHIFT_I64(ShrI64U, uint64_t, >>);

      //===----------------------------------------------------------------===//
      // ExtI64: Comparison ops
      //===----------------------------------------------------------------===//

#define DISPATCH_OP_EXT_I64_CMP_I64(op_name, type, op) \
  DISPATCH_OP(EXT_I64, op_name, {                      \
    int64_t lhs = VM_DecOperandRegI64("lhs");          \
    int64_t rhs = VM_DecOperandRegI64("rhs");          \
    int32_t* result = VM_DecResultRegI32("result");    \
    *result = (((type)lhs)op((type)rhs)) ? 1 : 0;      \
  });

      DISPATCH_OP_EXT_I64_CMP_I64(CmpEQI64, int64_t, ==);
      DISPATCH_OP_EXT_I64_CMP_I64(CmpNEI64, int64_t, !=);
      DISPATCH_OP_EXT_I64_CMP_I64(CmpLTI64S, int64_t, <);
      DISPATCH_OP_EXT_I64_CMP_I64(CmpLTI64U, uint64_t, <);
      DISPATCH_OP(EXT_I64, CmpNZI64, {
        int64_t operand = VM_DecOperandRegI64("operand");
        int32_t* result = VM_DecResultRegI32("result");
        *result = (operand != 0) ? 1 : 0;
      });
#else
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
#endif  // IREE_VM_EXT_I64_ENABLE
    }
    END_DISPATCH_PREFIX();

    DISPATCH_OP(CORE, PrefixExtF32,
                { return iree_make_status(IREE_STATUS_UNIMPLEMENTED); });

    DISPATCH_OP(CORE, PrefixExtF64,
                { return iree_make_status(IREE_STATUS_UNIMPLEMENTED); });

    // NOLINTNEXTLINE(misc-static-assert)
    DISPATCH_UNHANDLED_CORE();
  }
  END_DISPATCH_CORE();
}
