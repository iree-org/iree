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

#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_dispatch_util.h"
#include "iree/vm/ops.h"

//===----------------------------------------------------------------------===//
// Register remapping utilities
//===----------------------------------------------------------------------===//

// Remaps registers from a source set to a destination set within the same stack
// frame. This is a way to perform a conditional multi-mov sequence instead of
// requiring the additional bytecode representation of the conditional movs.
//
// This assumes that the remapping list is properly ordered such that there are
// no swapping hazards (such as 0->1,1->0). The register allocator in the
// compiler should ensure this is the case when it can occur.
static void iree_vm_bytecode_dispatch_remap_branch_registers(
    const iree_vm_registers_t regs,
    const iree_vm_register_remap_list_t* IREE_RESTRICT remap_list) {
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
    const iree_vm_registers_t regs,
    const iree_vm_register_list_t* IREE_RESTRICT reg_list) {
  for (int i = 0; i < reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    uint16_t reg = reg_list->registers[i];
    if ((reg & (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) ==
        (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) {
      iree_vm_ref_release(&regs.ref[reg & regs.ref_mask]);
    }
  }
}

//===----------------------------------------------------------------------===//
// Stack management
//===----------------------------------------------------------------------===//

static iree_vm_registers_t iree_vm_bytecode_get_register_storage(
    iree_vm_stack_frame_t* frame) {
  const iree_vm_bytecode_frame_storage_t* stack_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(frame);

  // Masks indicate the valid bits of any register value within the range we
  // have allocated in the storage. So for 4 registers we'd expect a 0b11 mask.
  iree_vm_registers_t registers;
  memset(&registers, 0, sizeof(registers));
  registers.i32_mask = (uint16_t)(stack_storage->i32_register_count
                                      ? stack_storage->i32_register_count - 1
                                      : 0);
  registers.ref_mask = (uint16_t)(stack_storage->ref_register_count
                                      ? stack_storage->ref_register_count - 1
                                      : 0);

  // Register storage immediately follows the stack storage header.
  registers.i32 =
      (int32_t*)((uintptr_t)stack_storage + stack_storage->i32_register_offset);
  registers.ref = (iree_vm_ref_t*)((uintptr_t)stack_storage +
                                   stack_storage->ref_register_offset);

  return registers;
}

// Releases any remaining refs held in the frame storage.
static void IREE_API_CALL
iree_vm_bytecode_stack_frame_cleanup(iree_vm_stack_frame_t* frame) {
  iree_vm_registers_t regs = iree_vm_bytecode_get_register_storage(frame);
  // TODO(benvanik): allow the VM to elide this when it's known that there are
  // no more live registers.
  for (uint16_t i = 0; i <= regs.ref_mask; ++i) {
    iree_vm_ref_t* ref = &regs.ref[i];
    if (ref->ptr) iree_vm_ref_release(ref);
  }
}

static iree_status_t iree_vm_bytecode_function_enter(
    iree_vm_stack_t* stack, const iree_vm_function_t function,
    iree_vm_stack_frame_t** out_callee_frame,
    iree_vm_registers_t* out_callee_registers) {
  IREE_DISPATCH_LOG_CALL(&function);

  iree_vm_bytecode_module_t* module =
      (iree_vm_bytecode_module_t*)function.module->self;
  if (IREE_UNLIKELY(function.ordinal >= module->function_descriptor_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import ordinal out of range");
  }
  const iree_vm_FunctionDescriptor_t* target_descriptor =
      &module->function_descriptor_table[function.ordinal];

  // We first compute the frame size of the callee and the masks we'll use to
  // bounds check register access. This lets us allocate the entire frame
  // (header, frame, and register storage) as a single pointer bump below.

  // Round up register counts to the nearest power of 2 (if not already).
  // This let's us use bit masks on register accesses to do bounds checking
  // instead of more complex logic. The cost of these extra registers is only at
  // worst 2x the required cost: so not large when thinking about the normal
  // size of data used in an IREE app for tensors.
  //
  // Note that to allow the masking to work as a guard we need to ensure we at
  // least allocate 1 register; this way an i32[reg & mask] will always point at
  // valid memory even if mask == 0.
  uint32_t i32_register_count = iree_math_round_up_to_pow2_u32(
      VMMAX(1, target_descriptor->i32_register_count));
  uint32_t ref_register_count = iree_math_round_up_to_pow2_u32(
      VMMAX(1, target_descriptor->ref_register_count));
  if (IREE_UNLIKELY(i32_register_count > IREE_I32_REGISTER_MASK) ||
      IREE_UNLIKELY(ref_register_count > IREE_REF_REGISTER_MASK)) {
    // Register count overflow. A valid compiler should never produce files that
    // hit this.
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "register count overflow");
  }

  // We need to align the ref register start to the natural machine
  // alignment in case the compiler is expecting that (it makes it easier to
  // debug too).
  iree_host_size_t header_size =
      iree_math_align(sizeof(iree_vm_bytecode_frame_storage_t), 16);
  iree_host_size_t i32_register_size =
      iree_math_align(i32_register_count * sizeof(int32_t), 16);
  iree_host_size_t ref_register_size =
      iree_math_align(ref_register_count * sizeof(iree_vm_ref_t), 16);
  iree_host_size_t frame_size =
      header_size + i32_register_size + ref_register_size;

  // Enter function and allocate stack frame storage.
  IREE_RETURN_IF_ERROR(iree_vm_stack_function_enter(
      stack, &function, IREE_VM_STACK_FRAME_BYTECODE, frame_size,
      iree_vm_bytecode_stack_frame_cleanup, out_callee_frame));

  // Stash metadata and compute register pointers.
  iree_vm_bytecode_frame_storage_t* stack_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(
          *out_callee_frame);
  stack_storage->i32_register_count = i32_register_count;
  stack_storage->ref_register_count = ref_register_count;
  stack_storage->i32_register_offset = header_size;
  stack_storage->ref_register_offset = header_size + i32_register_size;
  *out_callee_registers =
      iree_vm_bytecode_get_register_storage(*out_callee_frame);

  return iree_ok_status();
}

// Enters an internal bytecode stack frame from an external caller.
// A new |out_callee_frame| will be pushed to the stack with storage space for
// the registers used by the function and |arguments| will be marshaled into the
// ABI-defined registers.
//
// Note that callers are expected to have matched our expectations for
// |arguments| and we don't validate that here.
static iree_status_t iree_vm_bytecode_external_enter(
    iree_vm_stack_t* stack, const iree_vm_function_t function,
    iree_string_view_t cconv_arguments, iree_byte_span_t arguments,
    iree_vm_stack_frame_t** out_callee_frame,
    iree_vm_registers_t* out_callee_registers) {
  // Enter the bytecode function and allocate registers.
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_enter(
      stack, function, out_callee_frame, out_callee_registers));

  // Marshal arguments from the ABI format to the VM registers.
  iree_vm_registers_t callee_registers = *out_callee_registers;
  uint16_t i32_reg = 0;
  uint16_t ref_reg = 0;
  const uint8_t* p = arguments.data;
  for (iree_host_size_t i = 0; i < cconv_arguments.size; ++i) {
    switch (cconv_arguments.data[i]) {
      case IREE_VM_CCONV_TYPE_INT32: {
        uint16_t dst_reg = i32_reg++;
        memcpy(&callee_registers.i32[dst_reg & callee_registers.i32_mask], p,
               sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_INT64: {
        uint16_t dst_reg = i32_reg;
        i32_reg += 2;
        memcpy(&callee_registers.i32[dst_reg & callee_registers.i32_mask], p,
               sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        uint16_t dst_reg = ref_reg++;
        iree_vm_ref_move(
            (iree_vm_ref_t*)p,
            &callee_registers.ref[dst_reg & callee_registers.ref_mask]);
        p += sizeof(iree_vm_ref_t);
      } break;
    }
  }

  return iree_ok_status();
}

// Leaves an internal bytecode stack frame and returns to an external caller.
// Registers will be marshaled from the |src_reg_list| to the |results| buffer.
//
// Note that callers are expected to have matched our expectations for
// |results| and we don't validate that here.
static iree_status_t iree_vm_bytecode_external_leave(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* callee_frame,
    const iree_vm_registers_t* IREE_RESTRICT callee_registers,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    iree_string_view_t cconv_results, iree_byte_span_t results) {
  // Marshal results from registers to the ABI results buffer.
  uint8_t* p = results.data;
  for (iree_host_size_t i = 0; i < cconv_results.size; ++i) {
    uint16_t src_reg = src_reg_list->registers[i];
    switch (cconv_results.data[i]) {
      case IREE_VM_CCONV_TYPE_INT32: {
        memcpy(p, &callee_registers->i32[src_reg & callee_registers->i32_mask],
               sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_INT64: {
        memcpy(
            p,
            &callee_registers->i32[src_reg & (callee_registers->i32_mask & ~1)],
            sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        iree_vm_ref_move(
            &callee_registers->ref[src_reg & callee_registers->ref_mask],
            (iree_vm_ref_t*)p);
        p += sizeof(iree_vm_ref_t);
      } break;
    }
  }

  // Leave and deallocate bytecode stack frame.
  return iree_vm_stack_function_leave(stack);
}

// Enters an internal bytecode stack frame from a parent bytecode frame.
// Registers in |src_reg_list| will be marshaled into the callee frame and the
// |dst_reg_list| will be stashed for use when leaving the frame.
static iree_status_t iree_vm_bytecode_internal_enter(
    iree_vm_stack_t* stack, iree_vm_module_t* module, int32_t function_ordinal,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    const iree_vm_register_list_t* IREE_RESTRICT dst_reg_list,
    iree_vm_stack_frame_t** out_callee_frame,
    iree_vm_registers_t* out_callee_registers) {
  // Stash the destination register list for result values on the caller.
  iree_vm_bytecode_frame_storage_t* caller_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(
          iree_vm_stack_current_frame(stack));
  caller_storage->return_registers = dst_reg_list;

  // NOTE: after this call the caller registers may be invalid and need to be
  // requeried.
  iree_vm_function_t function;
  function.module = module;
  function.linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
  function.ordinal = function_ordinal;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_enter(
      stack, function, out_callee_frame, out_callee_registers));

  // Remaps argument/result registers from a source list in the caller/callee
  // frame to the 0-N ABI registers in the callee/caller frame.
  // This assumes that the destination stack frame registers are unused and ok
  // to overwrite directly. Each bank begins left-aligned at 0 and increments
  // per arg of its type.
  iree_vm_registers_t src_regs =
      iree_vm_bytecode_get_register_storage(iree_vm_stack_parent_frame(stack));
  iree_vm_registers_t* dst_regs = out_callee_registers;
  int i32_reg_offset = 0;
  int ref_reg_offset = 0;
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      uint16_t dst_reg = ref_reg_offset++;
      memset(&dst_regs->ref[dst_reg & dst_regs->ref_mask], 0,
             sizeof(iree_vm_ref_t));
      iree_vm_ref_retain_or_move(src_reg & IREE_REF_REGISTER_MOVE_BIT,
                                 &src_regs.ref[src_reg & src_regs.ref_mask],
                                 &dst_regs->ref[dst_reg & dst_regs->ref_mask]);
    } else {
      uint16_t dst_reg = i32_reg_offset++;
      dst_regs->i32[dst_reg & dst_regs->i32_mask] =
          src_regs.i32[src_reg & src_regs.i32_mask];
    }
  }

  return iree_ok_status();
}

// Leaves an internal bytecode stack frame and returns to the parent bytecode
// frame. |src_reg_list| registers will be marshaled into the dst_reg_list
// provided by the caller frame when entering.
static iree_status_t iree_vm_bytecode_internal_leave(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* callee_frame,
    const iree_vm_registers_t callee_registers,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    iree_vm_stack_frame_t** out_caller_frame,
    iree_vm_registers_t* out_caller_registers) {
  // Remaps registers from source to destination across frames.
  // Registers from the |src_regs| will be copied/moved to |dst_regs| with the
  // mappings provided by |src_reg_list| and |dst_reg_list|. It's assumed that
  // the mappings are matching by type and - in the case that they aren't -
  // things will get weird (but not crash).
  *out_caller_frame = iree_vm_stack_parent_frame(stack);
  iree_vm_bytecode_frame_storage_t* caller_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(
          *out_caller_frame);
  const iree_vm_register_list_t* dst_reg_list =
      caller_storage->return_registers;
  VMCHECK(src_reg_list->size <= dst_reg_list->size);
  if (IREE_UNLIKELY(src_reg_list->size > dst_reg_list->size)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "src/dst reg count mismatch on internal return");
  }
  iree_vm_registers_t caller_registers =
      iree_vm_bytecode_get_register_storage(*out_caller_frame);
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    uint16_t dst_reg = dst_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(
          src_reg & IREE_REF_REGISTER_MOVE_BIT,
          &callee_registers.ref[src_reg & callee_registers.ref_mask],
          &caller_registers.ref[dst_reg & caller_registers.ref_mask]);
    } else {
      caller_registers.i32[dst_reg & caller_registers.i32_mask] =
          callee_registers.i32[src_reg & callee_registers.i32_mask];
    }
  }

  // Leave and deallocate bytecode stack frame.
  *out_caller_registers = caller_registers;
  return iree_vm_stack_function_leave(stack);
}

// Populates an import call arguments
static void iree_vm_bytecode_populate_import_cconv_arguments(
    iree_string_view_t cconv_arguments,
    const iree_vm_registers_t caller_registers,
    const iree_vm_register_list_t* IREE_RESTRICT segment_size_list,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    iree_byte_span_t storage) {
  uint8_t* IREE_RESTRICT p = storage.data;
  for (iree_host_size_t i = 0, seg_i = 0, reg_i = 0; i < cconv_arguments.size;
       ++i, ++seg_i) {
    switch (cconv_arguments.data[i]) {
      case IREE_VM_CCONV_TYPE_INT32: {
        memcpy(p,
               &caller_registers.i32[src_reg_list->registers[reg_i++] &
                                     caller_registers.i32_mask],
               sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_INT64: {
        memcpy(p,
               &caller_registers.i32[src_reg_list->registers[reg_i++] &
                                     (caller_registers.i32_mask & ~1)],
               sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        uint16_t src_reg = src_reg_list->registers[reg_i++];
        iree_vm_ref_retain_or_move(
            src_reg & IREE_REF_REGISTER_MOVE_BIT,
            &caller_registers.ref[src_reg & caller_registers.ref_mask],
            (iree_vm_ref_t*)p);
        p += sizeof(iree_vm_ref_t);
      } break;
      case IREE_VM_CCONV_TYPE_SPAN_START: {
        VMCHECK(segment_size_list);
        int32_t span_count = segment_size_list->registers[seg_i];
        memcpy(p, &span_count, sizeof(int32_t));
        p += sizeof(int32_t);
        if (!span_count) {
          // No items; skip the span.
          do {
            ++i;
          } while (i < cconv_arguments.size &&
                   cconv_arguments.data[i] != IREE_VM_CCONV_TYPE_SPAN_END);
          continue;
        }
        iree_host_size_t span_start_i = i + 1;
        for (int32_t j = 0; j < span_count; ++j) {
          for (i = span_start_i;
               i < cconv_arguments.size &&
               cconv_arguments.data[i] != IREE_VM_CCONV_TYPE_SPAN_END;
               ++i) {
            // TODO(benvanik): share with switch above.
            switch (cconv_arguments.data[i]) {
              case IREE_VM_CCONV_TYPE_INT32: {
                memcpy(p,
                       &caller_registers.i32[src_reg_list->registers[reg_i++] &
                                             caller_registers.i32_mask],
                       sizeof(int32_t));
                p += sizeof(int32_t);
              } break;
              case IREE_VM_CCONV_TYPE_INT64: {
                memcpy(p,
                       &caller_registers.i32[src_reg_list->registers[reg_i++] &
                                             (caller_registers.i32_mask & ~1)],
                       sizeof(int64_t));
                p += sizeof(int64_t);
              } break;
              case IREE_VM_CCONV_TYPE_REF: {
                uint16_t src_reg = src_reg_list->registers[reg_i++];
                iree_vm_ref_retain_or_move(
                    src_reg & IREE_REF_REGISTER_MOVE_BIT,
                    &caller_registers.ref[src_reg & caller_registers.ref_mask],
                    (iree_vm_ref_t*)p);
                p += sizeof(iree_vm_ref_t);
              } break;
            }
          }
        }
      } break;
    }
  }
}

// Issues a populated import call and marshals the results into |dst_reg_list|.
static iree_status_t iree_vm_bytecode_issue_import_call(
    iree_vm_stack_t* stack, const iree_vm_function_call_t call,
    iree_string_view_t cconv_results,
    const iree_vm_register_list_t* IREE_RESTRICT dst_reg_list,
    iree_vm_stack_frame_t** out_caller_frame,
    iree_vm_registers_t* out_caller_registers,
    iree_vm_execution_result_t* out_result) {
  // Call external function.
  iree_status_t call_status = call.function.module->begin_call(
      call.function.module->self, stack, &call, out_result);
  if (IREE_UNLIKELY(!iree_status_is_ok(call_status))) {
    // TODO(benvanik): set execution result to failure/capture stack.
    return iree_status_annotate(call_status,
                                iree_make_cstring_view("while calling import"));
  }

  // NOTE: we don't support yielding within imported functions right now so it's
  // safe to assume the stack is still valid here. If the called function can
  // yield then we'll need to requery all pointers here.
  *out_caller_frame = iree_vm_stack_current_frame(stack);
  *out_caller_registers =
      iree_vm_bytecode_get_register_storage(*out_caller_frame);

  // Marshal outputs from the ABI results buffer to registers.
  iree_vm_registers_t caller_registers = *out_caller_registers;
  uint8_t* IREE_RESTRICT p = call.results.data;
  for (iree_host_size_t i = 0; i < cconv_results.size && i < dst_reg_list->size;
       ++i) {
    uint16_t dst_reg = dst_reg_list->registers[i];
    switch (cconv_results.data[i]) {
      case IREE_VM_CCONV_TYPE_INT32:
        memcpy(&caller_registers.i32[dst_reg & caller_registers.i32_mask], p,
               sizeof(int32_t));
        p += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_INT64:
        memcpy(
            &caller_registers.i32[dst_reg & (caller_registers.i32_mask & ~1)],
            p, sizeof(int64_t));
        p += sizeof(int64_t);
        break;
      case IREE_VM_CCONV_TYPE_REF:
        iree_vm_ref_move(
            (iree_vm_ref_t*)p,
            &caller_registers.ref[dst_reg & caller_registers.ref_mask]);
        p += sizeof(iree_vm_ref_t);
        break;
    }
  }

  return iree_ok_status();
}

// Calls an imported function from another module.
// Marshals the |src_reg_list| registers into ABI storage and results into
// |dst_reg_list|.
static iree_status_t iree_vm_bytecode_call_import(
    iree_vm_stack_t* stack, const iree_vm_bytecode_module_state_t* module_state,
    uint32_t import_ordinal, const iree_vm_registers_t caller_registers,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    const iree_vm_register_list_t* IREE_RESTRICT dst_reg_list,
    iree_vm_stack_frame_t** out_caller_frame,
    iree_vm_registers_t* out_caller_registers,
    iree_vm_execution_result_t* out_result) {
  // Prepare |call| by looking up the import information.
  import_ordinal &= 0x7FFFFFFFu;
  if (IREE_UNLIKELY(import_ordinal >= module_state->import_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import ordinal out of range");
  }
  const iree_vm_bytecode_import_t* import =
      &module_state->import_table[import_ordinal];
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = import->function;
  IREE_DISPATCH_LOG_CALL(&call.function);

  // Marshal inputs from registers to the ABI arguments buffer.
  call.arguments.data_length = import->argument_buffer_size;
  call.arguments.data = iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);
  iree_vm_bytecode_populate_import_cconv_arguments(
      import->arguments, caller_registers,
      /*segment_size_list=*/NULL, src_reg_list, call.arguments);

  // Issue the call and handle results.
  call.results.data_length = import->result_buffer_size;
  call.results.data = iree_alloca(call.results.data_length);
  memset(call.results.data, 0, call.results.data_length);
  return iree_vm_bytecode_issue_import_call(stack, call, import->results,
                                            dst_reg_list, out_caller_frame,
                                            out_caller_registers, out_result);
}

// Calls a variadic imported function from another module.
// Marshals the |src_reg_list| registers into ABI storage and results into
// |dst_reg_list|. |segment_size_list| contains the counts within each segment.
static iree_status_t iree_vm_bytecode_call_import_variadic(
    iree_vm_stack_t* stack, const iree_vm_bytecode_module_state_t* module_state,
    uint32_t import_ordinal, const iree_vm_registers_t caller_registers,
    const iree_vm_register_list_t* IREE_RESTRICT segment_size_list,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    const iree_vm_register_list_t* IREE_RESTRICT dst_reg_list,
    iree_vm_stack_frame_t** out_caller_frame,
    iree_vm_registers_t* out_caller_registers,
    iree_vm_execution_result_t* out_result) {
  // Prepare |call| by looking up the import information.
  import_ordinal &= 0x7FFFFFFFu;
  if (IREE_UNLIKELY(import_ordinal >= module_state->import_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import ordinal out of range");
  }
  const iree_vm_bytecode_import_t* import =
      &module_state->import_table[import_ordinal];
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = import->function;
  IREE_DISPATCH_LOG_CALL(&call.function);

  // Allocate ABI argument/result storage taking into account the variadic
  // segments.
  IREE_RETURN_IF_ERROR(iree_vm_function_call_compute_cconv_fragment_size(
      import->arguments, segment_size_list, &call.arguments.data_length));
  call.arguments.data = iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);

  // Marshal inputs from registers to the ABI arguments buffer.
  iree_vm_bytecode_populate_import_cconv_arguments(
      import->arguments, caller_registers, segment_size_list, src_reg_list,
      call.arguments);

  // Issue the call and handle results.
  call.results.data_length = import->result_buffer_size;
  call.results.data = iree_alloca(call.results.data_length);
  memset(call.results.data, 0, call.results.data_length);
  return iree_vm_bytecode_issue_import_call(stack, call, import->results,
                                            dst_reg_list, out_caller_frame,
                                            out_caller_registers, out_result);
}

//===----------------------------------------------------------------------===//
// Main interpreter dispatch routine
//===----------------------------------------------------------------------===//

iree_status_t iree_vm_bytecode_dispatch(
    iree_vm_stack_t* stack, iree_vm_bytecode_module_t* module,
    const iree_vm_function_call_t* call, iree_string_view_t cconv_arguments,
    iree_string_view_t cconv_results, iree_vm_execution_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  // When required emit the dispatch tables here referencing the labels we are
  // defining below.
  DEFINE_DISPATCH_TABLES();

  // Enter function (as this is the initial call).
  // The callee's return will take care of storing the output registers when it
  // actually does return, either immediately or in the future via a resume.
  iree_vm_stack_frame_t* current_frame = NULL;
  iree_vm_registers_t regs;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_external_enter(stack, call->function, cconv_arguments,
                                      call->arguments, &current_frame, &regs));

  // Primary dispatch state. This is our 'native stack frame' and really
  // just enough to make dereferencing common addresses (like the current
  // offset) faster. You can think of this like CPU state (like PC).
  //
  // The hope is that the compiler decides to keep these in registers (as
  // they are touched for every instruction executed). The frame will change
  // as we call into different functions.
  const iree_vm_bytecode_module_state_t* IREE_RESTRICT module_state =
      (iree_vm_bytecode_module_state_t*)current_frame->module_state;
  const uint8_t* IREE_RESTRICT bytecode_data =
      module->bytecode_data.data +
      module->function_descriptor_table[current_frame->function.ordinal]
          .bytecode_offset;
  iree_vm_source_offset_t pc = current_frame->pc;
  const int32_t entry_frame_depth = current_frame->depth;

  BEGIN_DISPATCH_CORE() {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, GlobalLoadI32, {
      uint32_t byte_offset = VM_DecGlobalAttr("global");
      if (IREE_UNLIKELY(byte_offset >=
                        module_state->rwdata_storage.data_length)) {
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
      uint32_t byte_offset = VM_DecGlobalAttr("global");
      if (IREE_UNLIKELY(byte_offset >=
                        module_state->rwdata_storage.data_length)) {
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
      uint32_t byte_offset = VM_DecOperandRegI32("global");
      if (IREE_UNLIKELY(byte_offset >=
                        module_state->rwdata_storage.data_length)) {
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
      uint32_t byte_offset = VM_DecOperandRegI32("global");
      if (IREE_UNLIKELY(byte_offset >=
                        module_state->rwdata_storage.data_length)) {
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
      uint32_t global = VM_DecGlobalAttr("global");
      if (IREE_UNLIKELY(global >= module_state->global_ref_count)) {
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
      uint32_t global = VM_DecGlobalAttr("global");
      if (IREE_UNLIKELY(global >= module_state->global_ref_count)) {
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
      uint32_t global = VM_DecGlobalAttr("global");
      if (IREE_UNLIKELY(global >= module_state->global_ref_count)) {
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
      uint32_t global = VM_DecGlobalAttr("global");
      if (IREE_UNLIKELY(global >= module_state->global_ref_count)) {
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
      uint32_t rodata_ordinal = VM_DecRodataAttr("rodata");
      if (IREE_UNLIKELY(rodata_ordinal >= module_state->rodata_ref_count)) {
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
      uint32_t initial_capacity = VM_DecOperandRegI32("initial_capacity");
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
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t minimum_capacity = VM_DecOperandRegI32("minimum_capacity");
      IREE_RETURN_IF_ERROR(iree_vm_list_reserve(list, minimum_capacity));
    });

    DISPATCH_OP(CORE, ListSize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t* result = VM_DecResultRegI32("result");
      *result = (int32_t)iree_vm_list_size(list);
    });

    DISPATCH_OP(CORE, ListResize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t new_size = VM_DecOperandRegI32("new_size");
      IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, new_size));
    });

    DISPATCH_OP(CORE, ListGetI32, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t index = VM_DecOperandRegI32("index");
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
      uint32_t index = VM_DecOperandRegI32("index");
      int32_t raw_value = VM_DecOperandRegI32("raw_value");
      iree_vm_value_t value = iree_vm_value_make_i32(raw_value);
      IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
    });

    DISPATCH_OP(CORE, ListGetRef, {
      // bool list_is_move;
      // iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      // iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      // if (!list) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      // uint32_t index = VM_DecOperandRegI32("index");
      // iree_vm_ref_t* result = VM_DecResultRegRef("result");
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "vm.list.get.ref not implemented");
    });

    DISPATCH_OP(CORE, ListSetRef, {
      // bool list_is_move;
      // iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      // iree_vm_list_t* list = iree_vm_list_deref(list_ref);
      // if (!list) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      // uint32_t index = VM_DecOperandRegI32("index");
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

#define DISPATCH_OP_CORE_UNARY_ALU_I32(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                           \
    int32_t operand = VM_DecOperandRegI32("operand");    \
    int32_t* result = VM_DecResultRegI32("result");      \
    *result = op_func(operand);                          \
  });

#define DISPATCH_OP_CORE_BINARY_ALU_I32(op_name, op_func) \
  DISPATCH_OP(CORE, op_name, {                            \
    int32_t lhs = VM_DecOperandRegI32("lhs");             \
    int32_t rhs = VM_DecOperandRegI32("rhs");             \
    int32_t* result = VM_DecResultRegI32("result");       \
    *result = op_func(lhs, rhs);                          \
  });

    DISPATCH_OP_CORE_BINARY_ALU_I32(AddI32, vm_add_i32);
    DISPATCH_OP_CORE_BINARY_ALU_I32(SubI32, vm_sub_i32);
    DISPATCH_OP_CORE_BINARY_ALU_I32(MulI32, vm_mul_i32);
    DISPATCH_OP_CORE_BINARY_ALU_I32(DivI32S, vm_div_i32s);
    DISPATCH_OP_CORE_BINARY_ALU_I32(DivI32U, vm_div_i32u);
    DISPATCH_OP_CORE_BINARY_ALU_I32(RemI32S, vm_rem_i32s);
    DISPATCH_OP_CORE_BINARY_ALU_I32(RemI32U, vm_rem_i32u);
    DISPATCH_OP_CORE_UNARY_ALU_I32(NotI32, vm_not_i32);
    DISPATCH_OP_CORE_BINARY_ALU_I32(AndI32, vm_and_i32);
    DISPATCH_OP_CORE_BINARY_ALU_I32(OrI32, vm_or_i32);
    DISPATCH_OP_CORE_BINARY_ALU_I32(XorI32, vm_xor_i32);

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

#define DISPATCH_OP_CORE_SHIFT_I32(op_name, op_func)  \
  DISPATCH_OP(CORE, op_name, {                        \
    int32_t operand = VM_DecOperandRegI32("operand"); \
    int8_t amount = VM_DecConstI8("amount");          \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = op_func(operand, amount);               \
  });

    DISPATCH_OP_CORE_SHIFT_I32(ShlI32, vm_shl_i32);
    DISPATCH_OP_CORE_SHIFT_I32(ShrI32S, vm_shr_i32s);
    DISPATCH_OP_CORE_SHIFT_I32(ShrI32U, vm_shr_i32u);

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
        // Call import (and possible yield).
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_import(
            stack, module_state, function_ordinal, regs, src_reg_list,
            dst_reg_list, &current_frame, &regs, out_result));
      } else {
        // Switch execution to the target function and continue running in the
        // bytecode dispatcher.
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_internal_enter(
            stack, current_frame->function.module, function_ordinal,
            src_reg_list, dst_reg_list, &current_frame, &regs));
        bytecode_data =
            module->bytecode_data.data +
            module->function_descriptor_table[function_ordinal].bytecode_offset;
        pc = current_frame->pc;
      }
    });

    DISPATCH_OP(CORE, CallVariadic, {
      // TODO(benvanik): dedupe with above or merge and always have the seg size
      // list be present (but empty) for non-variadic calls.
      int32_t function_ordinal = VM_DecFuncAttr("callee");
      const iree_vm_register_list_t* segment_size_list =
          VM_DecVariadicOperands("segment_sizes");
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      const iree_vm_register_list_t* dst_reg_list =
          VM_DecVariadicResults("results");
      current_frame->pc = pc;

      // NOTE: we assume validation has ensured these functions exist.
      // TODO(benvanik): something more clever than just a high bit?
      int is_import = (function_ordinal & 0x80000000u) != 0;
      if (IREE_UNLIKELY(!is_import)) {
        // Variadic calls are currently only supported for import functions.
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "variadic calls only supported for internal callees");
      }

      // Call import (and possible yield).
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_import_variadic(
          stack, module_state, function_ordinal, regs, segment_size_list,
          src_reg_list, dst_reg_list, &current_frame, &regs, out_result));
    });

    DISPATCH_OP(CORE, Return, {
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      current_frame->pc = pc;

      if (current_frame->depth <= entry_frame_depth) {
        // Return from the top-level entry frame - return back to call().
        return iree_vm_bytecode_external_leave(stack, current_frame, &regs,
                                               src_reg_list, cconv_results,
                                               call->results);
      }

      // Store results into the caller frame and pop back to the parent.
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_internal_leave(
          stack, current_frame, regs, src_reg_list, &current_frame, &regs));

      // Reset dispatch state so we can continue executing in the caller.
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
        uint32_t byte_offset = VM_DecGlobalAttr("global");
        if (IREE_UNLIKELY(byte_offset >=
                          module_state->rwdata_storage.data_length)) {
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
        uint32_t byte_offset = VM_DecGlobalAttr("global");
        if (IREE_UNLIKELY(byte_offset >=
                          module_state->rwdata_storage.data_length)) {
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
        uint32_t byte_offset = VM_DecOperandRegI32("global");
        if (IREE_UNLIKELY(byte_offset >=
                          module_state->rwdata_storage.data_length)) {
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
        uint32_t byte_offset = VM_DecOperandRegI32("global");
        if (IREE_UNLIKELY(byte_offset >=
                          module_state->rwdata_storage.data_length)) {
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
        if (IREE_UNLIKELY(!list)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
        }
        uint32_t index = VM_DecOperandRegI32("index");
        int64_t* result = VM_DecResultRegI64("result");
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            list, index, IREE_VM_VALUE_TYPE_I64, &value));
        *result = value.i64;
      });

      DISPATCH_OP(EXT_I64, ListSetI64, {
        bool list_is_move;
        iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
        iree_vm_list_t* list = iree_vm_list_deref(list_ref);
        if (IREE_UNLIKELY(!list)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
        }
        uint32_t index = VM_DecOperandRegI32("index");
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
