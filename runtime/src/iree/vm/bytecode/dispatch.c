// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/disassembler.h"
#include "iree/vm/bytecode/dispatch_util.h"
#include "iree/vm/bytecode/module_impl.h"
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
    int32_t* IREE_RESTRICT regs_i32, iree_vm_ref_t* IREE_RESTRICT regs_ref,
    const iree_vm_register_remap_list_t* IREE_RESTRICT remap_list) {
  for (int i = 0; i < remap_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = remap_list->pairs[i].src_reg;
    uint16_t dst_reg = remap_list->pairs[i].dst_reg;
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(src_reg & IREE_REF_REGISTER_MOVE_BIT,
                                 &regs_ref[src_reg & IREE_REF_REGISTER_MASK],
                                 &regs_ref[dst_reg & IREE_REF_REGISTER_MASK]);
    } else {
      regs_i32[dst_reg] = regs_i32[src_reg];
    }
  }
}

// Discards ref registers in the list if they are marked move.
// This can be used to eagerly release resources we don't need and reduces
// memory consumption if used effectively prior to yields/waits.
static void iree_vm_bytecode_dispatch_discard_registers(
    iree_vm_ref_t* IREE_RESTRICT regs_ref,
    const iree_vm_register_list_t* IREE_RESTRICT reg_list) {
  for (int i = 0; i < reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    uint16_t reg = reg_list->registers[i];
    if ((reg & (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) ==
        (IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT)) {
      iree_vm_ref_release(&regs_ref[reg & IREE_REF_REGISTER_MASK]);
    }
  }
}

//===----------------------------------------------------------------------===//
// Stack management
//===----------------------------------------------------------------------===//

static inline iree_vm_registers_t iree_vm_bytecode_get_register_storage(
    iree_vm_stack_frame_t* frame) {
  const iree_vm_bytecode_frame_storage_t* stack_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(frame);
  return (iree_vm_registers_t){
      .i32 = (int32_t*)((uintptr_t)stack_storage +
                        stack_storage->i32_register_offset),
      .ref = (iree_vm_ref_t*)((uintptr_t)stack_storage +
                              stack_storage->ref_register_offset),
  };
}

// Releases any remaining refs held in the frame storage.
static void iree_vm_bytecode_stack_frame_cleanup(iree_vm_stack_frame_t* frame) {
  // TODO(benvanik): allow the VM to elide this when it's known that there are
  // no more live registers.
  const iree_vm_bytecode_frame_storage_t* stack_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(frame);
  iree_vm_ref_t* refs = (iree_vm_ref_t*)((uintptr_t)stack_storage +
                                         stack_storage->ref_register_offset);
  for (uint16_t i = 0; i < stack_storage->ref_register_count; ++i) {
    iree_vm_ref_t* ref = &refs[i];
    if (ref->ptr) iree_vm_ref_release(ref);
  }
}

static iree_status_t iree_vm_bytecode_function_enter(
    iree_vm_stack_t* stack, const iree_vm_function_t function,
    iree_string_view_t cconv_results,
    iree_vm_stack_frame_t* IREE_RESTRICT* out_callee_frame,
    iree_vm_registers_t* out_callee_registers) {
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

  // We've verified all register storage prior to execution.
  uint32_t i32_register_count = target_descriptor->i32_register_count;
  uint32_t ref_register_count = target_descriptor->ref_register_count;
  IREE_ASSERT_LE(i32_register_count, IREE_I32_REGISTER_MASK);
  IREE_ASSERT_LE(ref_register_count, IREE_REF_REGISTER_MASK);

  // We need to align the ref register start to the natural machine
  // alignment in case the compiler is expecting that (it makes it easier to
  // debug too).
  iree_host_size_t header_size =
      iree_host_align(sizeof(iree_vm_bytecode_frame_storage_t), 16);
  iree_host_size_t i32_register_size =
      iree_host_align(i32_register_count * sizeof(int32_t), 16);
  iree_host_size_t ref_register_size =
      iree_host_align(ref_register_count * sizeof(iree_vm_ref_t), 16);
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
  stack_storage->cconv_results = cconv_results;
  stack_storage->i32_register_count = i32_register_count;
  stack_storage->i32_register_offset = header_size;
  stack_storage->ref_register_count = ref_register_count;
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
    iree_string_view_t cconv_results,
    iree_vm_stack_frame_t* IREE_RESTRICT* out_callee_frame,
    iree_vm_registers_t* out_callee_registers) {
  // Enter the bytecode function and allocate registers.
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_enter(
      stack, function, cconv_results, out_callee_frame, out_callee_registers));

  // Marshal arguments from the ABI format to the VM registers.
  iree_vm_registers_t callee_registers = *out_callee_registers;
  uint16_t i32_reg = 0;
  uint16_t ref_reg = 0;
  const uint8_t* p = arguments.data;
  for (iree_host_size_t i = 0; i < cconv_arguments.size; ++i) {
    switch (cconv_arguments.data[i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32: {
        uint16_t dst_reg = i32_reg++;
        memcpy(&callee_registers.i32[dst_reg], p, sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64: {
        uint16_t dst_reg = i32_reg;
        i32_reg += 2;
        memcpy(&callee_registers.i32[dst_reg], p, sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        uint16_t dst_reg = ref_reg++;
        iree_vm_ref_move(
            (iree_vm_ref_t*)p,
            &callee_registers.ref[dst_reg & IREE_REF_REGISTER_MASK]);
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
    iree_byte_span_t results) {
  const iree_vm_bytecode_frame_storage_t* stack_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(
          callee_frame);

  // Marshal results from registers to the ABI results buffer.
  iree_string_view_t cconv_results = stack_storage->cconv_results;
  uint8_t* p = results.data;
  for (iree_host_size_t i = 0; i < cconv_results.size; ++i) {
    uint16_t src_reg = src_reg_list->registers[i];
    switch (cconv_results.data[i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32: {
        memcpy(p, &callee_registers->i32[src_reg], sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64: {
        memcpy(p, &callee_registers->i32[src_reg], sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        iree_vm_ref_retain_or_move(
            src_reg & IREE_REF_REGISTER_MOVE_BIT,
            &callee_registers->ref[src_reg & IREE_REF_REGISTER_MASK],
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
    iree_vm_stack_frame_t* IREE_RESTRICT* out_callee_frame,
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
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_function_enter(stack, function, iree_string_view_empty(),
                                      out_callee_frame, out_callee_registers));

  // Remaps argument/result registers from a source list in the caller/callee
  // frame to the 0-N ABI registers in the callee/caller frame.
  // This assumes that the destination stack frame registers are unused and ok
  // to overwrite directly. Each bank begins left-aligned at 0 and increments
  // per arg of its type.
  iree_vm_stack_frame_t* caller_frame = iree_vm_stack_parent_frame(stack);
  IREE_ASSERT(caller_frame);
  iree_vm_registers_t src_regs =
      iree_vm_bytecode_get_register_storage(caller_frame);
  iree_vm_registers_t* dst_regs = out_callee_registers;
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
          &src_regs.ref[src_reg & IREE_REF_REGISTER_MASK],
          &dst_regs->ref[dst_reg & IREE_REF_REGISTER_MASK]);
    } else {
      uint16_t dst_reg = i32_reg_offset++;
      dst_regs->i32[dst_reg] = src_regs.i32[src_reg];
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
    iree_vm_stack_frame_t* IREE_RESTRICT* out_caller_frame,
    iree_vm_registers_t* out_caller_registers) {
  // Remaps registers from source to destination across frames.
  // Registers from the |src_regs| will be copied/moved to |dst_regs| with the
  // mappings provided by |src_reg_list| and |dst_reg_list|. It's assumed that
  // the mappings are matching by type and - in the case that they aren't -
  // things will get weird (but not crash).
  iree_vm_stack_frame_t* caller_frame = iree_vm_stack_parent_frame(stack);
  if (IREE_UNLIKELY(!caller_frame)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "unbalanced internal leave stack; stack root cannot not be internal");
  }
  iree_vm_bytecode_frame_storage_t* caller_storage =
      (iree_vm_bytecode_frame_storage_t*)iree_vm_stack_frame_storage(
          caller_frame);
  const iree_vm_register_list_t* dst_reg_list =
      caller_storage->return_registers;
  IREE_ASSERT_LE(src_reg_list->size, dst_reg_list->size);
  if (IREE_UNLIKELY(src_reg_list->size > dst_reg_list->size)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "src/dst reg count mismatch on internal return");
  }
  iree_vm_registers_t caller_registers =
      iree_vm_bytecode_get_register_storage(caller_frame);
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    uint16_t dst_reg = dst_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(
          src_reg & IREE_REF_REGISTER_MOVE_BIT,
          &callee_registers.ref[src_reg & IREE_REF_REGISTER_MASK],
          &caller_registers.ref[dst_reg & IREE_REF_REGISTER_MASK]);
    } else {
      caller_registers.i32[dst_reg] = callee_registers.i32[src_reg];
    }
  }

  // Leave and deallocate bytecode stack frame.
  *out_caller_frame = caller_frame;
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
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32: {
        memcpy(p, &caller_registers.i32[src_reg_list->registers[reg_i++]],
               sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64: {
        memcpy(p, &caller_registers.i32[src_reg_list->registers[reg_i++]],
               sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        uint16_t src_reg = src_reg_list->registers[reg_i++];
        iree_vm_ref_assign(
            &caller_registers.ref[src_reg & IREE_REF_REGISTER_MASK],
            (iree_vm_ref_t*)p);
        p += sizeof(iree_vm_ref_t);
      } break;
      case IREE_VM_CCONV_TYPE_SPAN_START: {
        IREE_ASSERT(segment_size_list);
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
              case IREE_VM_CCONV_TYPE_VOID:
                break;
              case IREE_VM_CCONV_TYPE_I32:
              case IREE_VM_CCONV_TYPE_F32: {
                memcpy(p,
                       &caller_registers.i32[src_reg_list->registers[reg_i++]],
                       sizeof(int32_t));
                p += sizeof(int32_t);
              } break;
              case IREE_VM_CCONV_TYPE_I64:
              case IREE_VM_CCONV_TYPE_F64: {
                memcpy(p,
                       &caller_registers.i32[src_reg_list->registers[reg_i++]],
                       sizeof(int64_t));
                p += sizeof(int64_t);
              } break;
              case IREE_VM_CCONV_TYPE_REF: {
                uint16_t src_reg = src_reg_list->registers[reg_i++];
                iree_vm_ref_assign(
                    &caller_registers.ref[src_reg & IREE_REF_REGISTER_MASK],
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
    iree_vm_stack_frame_t* IREE_RESTRICT* out_caller_frame,
    iree_vm_registers_t* out_caller_registers) {
  // Call external function.
  iree_status_t call_status =
      call.function.module->begin_call(call.function.module->self, stack, call);
  if (iree_status_is_deferred(call_status)) {
    if (!iree_byte_span_is_empty(call.results)) {
      iree_status_ignore(call_status);
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "yield in imports with results not supported");
    }
    return call_status;  // deferred for future resume
  } else if (IREE_UNLIKELY(!iree_status_is_ok(call_status))) {
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
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
        memcpy(&caller_registers.i32[dst_reg], p, sizeof(int32_t));
        p += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
        memcpy(&caller_registers.i32[dst_reg], p, sizeof(int64_t));
        p += sizeof(int64_t);
        break;
      case IREE_VM_CCONV_TYPE_REF:
        iree_vm_ref_move(
            (iree_vm_ref_t*)p,
            &caller_registers.ref[dst_reg & IREE_REF_REGISTER_MASK]);
        p += sizeof(iree_vm_ref_t);
        break;
    }
  }

  return iree_ok_status();
}

// Verifies that the requested import is valid and returns its table entry.
static iree_status_t iree_vm_bytecode_verify_import(
    iree_vm_stack_t* stack, const iree_vm_bytecode_module_state_t* module_state,
    uint32_t import_ordinal, const iree_vm_bytecode_import_t** out_import) {
  *out_import = NULL;

  // Ordinal has been checked as in-bounds during verification.
  import_ordinal &= 0x7FFFFFFFu;
  IREE_ASSERT(import_ordinal < module_state->import_count);

  const iree_vm_bytecode_import_t* import =
      &module_state->import_table[import_ordinal];
  if (!import->function.module) {
#if IREE_STATUS_MODE
    iree_vm_function_t decl_function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        iree_vm_stack_current_frame(stack)->function.module,
        IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL, import_ordinal,
        &decl_function));
    iree_string_view_t import_name = iree_vm_function_name(&decl_function);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "optional import `%.*s` (ordinal %u) not resolved",
                            (int)import_name.size, import_name.data,
                            import_ordinal);
#else
    return iree_make_status(IREE_STATUS_NOT_FOUND);
#endif  // IREE_STATUS_MODE
  }

  *out_import = import;
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
    iree_vm_stack_frame_t* IREE_RESTRICT* out_caller_frame,
    iree_vm_registers_t* out_caller_registers) {
  // Prepare |call| by looking up the import information.
  const iree_vm_bytecode_import_t* import = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_import(stack, module_state,
                                                      import_ordinal, &import));

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = import->function;

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
                                            out_caller_registers);
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
    iree_vm_stack_frame_t* IREE_RESTRICT* out_caller_frame,
    iree_vm_registers_t* out_caller_registers) {
  // Prepare |call| by looking up the import information.
  const iree_vm_bytecode_import_t* import = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_import(stack, module_state,
                                                      import_ordinal, &import));

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = import->function;

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
                                            out_caller_registers);
}

//===----------------------------------------------------------------------===//
// Main interpreter dispatch routine
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_bytecode_dispatch(
    iree_vm_stack_t* stack, iree_vm_bytecode_module_t* module,
    iree_vm_stack_frame_t* current_frame, iree_vm_registers_t regs,
    iree_byte_span_t call_results);

iree_status_t iree_vm_bytecode_dispatch_begin(
    iree_vm_stack_t* stack, iree_vm_bytecode_module_t* module,
    const iree_vm_function_call_t call, iree_string_view_t cconv_arguments,
    iree_string_view_t cconv_results) {
  // Enter function (as this is the initial call).
  // The callee's return will take care of storing the output registers when it
  // actually does return, either immediately or in the future via a resume.
  iree_vm_stack_frame_t* current_frame = NULL;
  iree_vm_registers_t regs;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_external_enter(
      stack, call.function, cconv_arguments, call.arguments, cconv_results,
      &current_frame, &regs));

  return iree_vm_bytecode_dispatch(stack, module, current_frame, regs,
                                   call.results);
}

iree_status_t iree_vm_bytecode_dispatch_resume(
    iree_vm_stack_t* stack, iree_vm_bytecode_module_t* module,
    iree_byte_span_t call_results) {
  iree_vm_stack_frame_t* current_frame = iree_vm_stack_top(stack);
  if (IREE_UNLIKELY(!current_frame)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no frame at top of stack to resume");
  }
  iree_vm_registers_t regs =
      iree_vm_bytecode_get_register_storage(current_frame);
  // TODO(benvanik): assert the module is at the top of the frame? We should
  // only be coming in from a call based on the current frame.
  return iree_vm_bytecode_dispatch(stack, module, current_frame, regs,
                                   call_results);
}

static iree_status_t iree_vm_bytecode_dispatch(
    iree_vm_stack_t* IREE_RESTRICT stack,
    iree_vm_bytecode_module_t* IREE_RESTRICT module,
    iree_vm_stack_frame_t* IREE_RESTRICT current_frame,
    iree_vm_registers_t regs, iree_byte_span_t call_results) {
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
  const iree_vm_bytecode_module_state_t* IREE_RESTRICT module_state =
      (iree_vm_bytecode_module_state_t*)current_frame->module_state;
  const uint8_t* IREE_RESTRICT bytecode_data =
      module->bytecode_data.data +
      module->function_descriptor_table[current_frame->function.ordinal]
          .bytecode_offset;

  int32_t* IREE_RESTRICT regs_i32 = regs.i32;
  IREE_BUILTIN_ASSUME_ALIGNED(regs_i32, 16);
  iree_vm_ref_t* IREE_RESTRICT regs_ref = regs.ref;
  IREE_BUILTIN_ASSUME_ALIGNED(regs_ref, 16);

  iree_vm_source_offset_t pc = current_frame->pc;
  BEGIN_DISPATCH_CORE() {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, GlobalLoadI32, {
      uint32_t byte_offset = VM_DecGlobalAttr("global");
      IREE_ASSERT(byte_offset + 4 <= module_state->rwdata_storage.data_length);
      int32_t* value = VM_DecResultRegI32("value");
      const int32_t global_value =
          vm_global_load_i32(module_state->rwdata_storage.data, byte_offset);
      *value = global_value;
    });

    DISPATCH_OP(CORE, GlobalStoreI32, {
      uint32_t byte_offset = VM_DecGlobalAttr("global");
      IREE_ASSERT(byte_offset + 4 <= module_state->rwdata_storage.data_length);
      int32_t value = VM_DecOperandRegI32("value");
      vm_global_store_i32(module_state->rwdata_storage.data, byte_offset,
                          value);
    });

    DISPATCH_OP(CORE, GlobalLoadIndirectI32, {
      uint32_t byte_offset = VM_DecOperandRegI32("global");
      if (IREE_UNLIKELY(byte_offset + 4 >
                        module_state->rwdata_storage.data_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int32_t* value = VM_DecResultRegI32("value");
      const int32_t global_value =
          vm_global_load_i32(module_state->rwdata_storage.data, byte_offset);
      *value = global_value;
    });

    DISPATCH_OP(CORE, GlobalStoreIndirectI32, {
      uint32_t byte_offset = VM_DecOperandRegI32("global");
      if (IREE_UNLIKELY(byte_offset + 4 >
                        module_state->rwdata_storage.data_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int32_t value = VM_DecOperandRegI32("value");
      vm_global_store_i32(module_state->rwdata_storage.data, byte_offset,
                          value);
    });

    DISPATCH_OP(CORE, GlobalLoadI64, {
      uint32_t byte_offset = VM_DecGlobalAttr("global");
      IREE_ASSERT(byte_offset + 8 <= module_state->rwdata_storage.data_length);
      int64_t* value = VM_DecResultRegI64("value");
      const int64_t global_value =
          vm_global_load_i64(module_state->rwdata_storage.data, byte_offset);
      *value = global_value;
    });

    DISPATCH_OP(CORE, GlobalStoreI64, {
      uint32_t byte_offset = VM_DecGlobalAttr("global");
      IREE_ASSERT(byte_offset + 8 <= module_state->rwdata_storage.data_length);
      int64_t value = VM_DecOperandRegI64("value");
      vm_global_store_i64(module_state->rwdata_storage.data, byte_offset,
                          value);
    });

    DISPATCH_OP(CORE, GlobalLoadIndirectI64, {
      uint32_t byte_offset = VM_DecOperandRegI32("global");
      if (IREE_UNLIKELY(byte_offset + 8 >
                        module_state->rwdata_storage.data_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int64_t* value = VM_DecResultRegI64("value");
      const int64_t global_value =
          vm_global_load_i64(module_state->rwdata_storage.data, byte_offset);
      *value = global_value;
    });

    DISPATCH_OP(CORE, GlobalStoreIndirectI64, {
      uint32_t byte_offset = VM_DecOperandRegI32("global");
      if (IREE_UNLIKELY(byte_offset + 8 >
                        module_state->rwdata_storage.data_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
            module_state->rwdata_storage.data_length);
      }
      int64_t value = VM_DecOperandRegI64("value");
      vm_global_store_i64(module_state->rwdata_storage.data, byte_offset,
                          value);
    });

    DISPATCH_OP(CORE, GlobalLoadRef, {
      uint32_t global = VM_DecGlobalAttr("global");
      IREE_ASSERT(global < module_state->global_ref_count);
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          result_is_move, global_ref, type_def->ref_type, result));
    });

    DISPATCH_OP(CORE, GlobalStoreRef, {
      uint32_t global = VM_DecGlobalAttr("global");
      IREE_ASSERT(global < module_state->global_ref_count);
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("value");
      bool value_is_move;
      iree_vm_ref_t* value = VM_DecOperandRegRef("value", &value_is_move);
      iree_vm_ref_t* global_ref = &module_state->global_ref_table[global];
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          value_is_move, value, type_def->ref_type, global_ref));
    });

    DISPATCH_OP(CORE, GlobalLoadIndirectRef, {
      uint32_t global = VM_DecOperandRegI32("global");
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
      uint32_t global = VM_DecOperandRegI32("global");
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

    DISPATCH_OP(CORE, ConstI64, {
      int64_t value = VM_DecIntAttr64("value");
      int64_t* result = VM_DecResultRegI64("result");
      *result = value;
    });

    DISPATCH_OP(CORE, ConstI64Zero, {
      int64_t* result = VM_DecResultRegI64("result");
      *result = 0;
    });

    DISPATCH_OP(CORE, ConstRefZero, {
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_ref_release(result);
    });

    DISPATCH_OP(CORE, ConstRefRodata, {
      uint32_t rodata_ordinal = VM_DecRodataAttr("rodata");
      IREE_ASSERT(rodata_ordinal < module_state->rodata_ref_count);
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("value", &result_is_move);
      IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_retain(
          &module_state->rodata_ref_table[rodata_ordinal],
          iree_vm_buffer_type_id(), result));
    });

    //===------------------------------------------------------------------===//
    // Buffers
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, BufferAlloc, {
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      bool result_is_move;
      iree_vm_ref_t* result_ref = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_buffer_t* buffer = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
          IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST,
          length, module_state->allocator, &buffer));
      IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
          buffer, iree_vm_buffer_type_id(), result_ref));
    });

    DISPATCH_OP(CORE, BufferClone, {
      bool source_is_move;
      iree_vm_ref_t* source_ref =
          VM_DecOperandRegRef("source", &source_is_move);
      iree_vm_buffer_t* source = iree_vm_buffer_deref(*source_ref);
      if (IREE_UNLIKELY(!source)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "source is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      bool result_is_move;
      iree_vm_ref_t* result_ref = VM_DecResultRegRef("result", &result_is_move);
      iree_vm_buffer_t* result = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
          IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST,
          source, offset, length, module_state->allocator, &result));
      IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
          result, iree_vm_buffer_type_id(), result_ref));
    });

    DISPATCH_OP(CORE, BufferLength, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
      }
      uint64_t* result = VM_DecResultRegI64("result");
      *result = (uint64_t)iree_vm_buffer_length(buffer);
    });

    DISPATCH_OP(CORE, BufferCopy, {
      bool source_buffer_is_move;
      iree_vm_ref_t* source_buffer_ref =
          VM_DecOperandRegRef("source_buffer", &source_buffer_is_move);
      iree_vm_buffer_t* source_buffer =
          iree_vm_buffer_deref(*source_buffer_ref);
      if (IREE_UNLIKELY(!source_buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t source_offset =
          VM_DecOperandRegI64HostSize("source_offset");
      bool target_buffer_is_move;
      iree_vm_ref_t* target_buffer_ref =
          VM_DecOperandRegRef("target_buffer", &target_buffer_is_move);
      iree_vm_buffer_t* target_buffer =
          iree_vm_buffer_deref(*target_buffer_ref);
      if (IREE_UNLIKELY(!target_buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "target_buffer is null");
      }
      iree_host_size_t target_offset =
          VM_DecOperandRegI64HostSize("target_offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      IREE_RETURN_IF_ERROR(iree_vm_buffer_copy_bytes(
          source_buffer, source_offset, target_buffer, target_offset, length));
    });

    DISPATCH_OP(CORE, BufferCompare, {
      bool lhs_buffer_is_move;
      iree_vm_ref_t* lhs_buffer_ref =
          VM_DecOperandRegRef("lhs_buffer", &lhs_buffer_is_move);
      iree_vm_buffer_t* lhs_buffer = iree_vm_buffer_deref(*lhs_buffer_ref);
      if (IREE_UNLIKELY(!lhs_buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "lhs_buffer is null");
      }
      iree_host_size_t lhs_offset = VM_DecOperandRegI64HostSize("lhs_offset");
      bool rhs_buffer_is_move;
      iree_vm_ref_t* rhs_buffer_ref =
          VM_DecOperandRegRef("rhs_buffer", &rhs_buffer_is_move);
      iree_vm_buffer_t* rhs_buffer = iree_vm_buffer_deref(*rhs_buffer_ref);
      if (IREE_UNLIKELY(!rhs_buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rhs_buffer is null");
      }
      iree_host_size_t rhs_offset = VM_DecOperandRegI64HostSize("rhs_offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      int32_t* result = VM_DecResultRegI32("result");
      IREE_RETURN_IF_ERROR(vm_buffer_compare(lhs_buffer, lhs_offset, rhs_buffer,
                                             rhs_offset, length, result));
    });

    // TODO(benvanik): rework dispatch so that the FillI* ops can share the same
    // body - they all only vary by the length passed to fill_elements. The
    // gotcha is that on big-endian machines we'd have to flip around the bytes.
    // See VMOpcodesCore.td for more information on the encoding.
    DISPATCH_OP(CORE, BufferFillI8, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      uint8_t value = (uint8_t)VM_DecOperandRegI32("value");
      vm_buffer_fill_i8_inline(buffer, offset, length, value);
    });
    DISPATCH_OP(CORE, BufferFillI16, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      uint16_t value = (uint16_t)VM_DecOperandRegI32("value");
      vm_buffer_fill_i16_inline(buffer, offset, length, value);
    });
    DISPATCH_OP(CORE, BufferFillI32, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      uint32_t value = VM_DecOperandRegI32("value");
      vm_buffer_fill_i32_inline(buffer, offset, length, value);
    });
    DISPATCH_OP(CORE, BufferFillI64, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
      uint64_t value = VM_DecOperandRegI64("value");
      vm_buffer_fill_i64_inline(buffer, offset, length, value);
    });

    // TODO(benvanik): rework dispatch so that the LoadI* ops can share the same
    // body - they only vary on the length and sign/zero extension mode but
    // can be packed into a single handler to reduce code-size.
    // See VMOpcodesCore.td for more information on the encoding.
    DISPATCH_OP(CORE, BufferLoadI8U, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("source_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
      uint32_t* result = VM_DecResultRegI32("result");
      vm_buffer_load_i8u_inline(buffer, offset, result);
    });
    DISPATCH_OP(CORE, BufferLoadI8S, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("source_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
      uint32_t* result = VM_DecResultRegI32("result");
      vm_buffer_load_i8s_inline(buffer, offset, result);
    });
    DISPATCH_OP(CORE, BufferLoadI16U, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("source_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
      uint32_t* result = VM_DecResultRegI32("result");
      vm_buffer_load_i16u_inline(buffer, offset, result);
    });
    DISPATCH_OP(CORE, BufferLoadI16S, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("source_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
      uint32_t* result = VM_DecResultRegI32("result");
      vm_buffer_load_i16s_inline(buffer, offset, result);
    });
    DISPATCH_OP(CORE, BufferLoadI32, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("source_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
      uint32_t* result = VM_DecResultRegI32("result");
      vm_buffer_load_i32_inline(buffer, offset, result);
    });
    DISPATCH_OP(CORE, BufferLoadI64, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("source_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
      uint64_t* result = VM_DecResultRegI64("result");
      vm_buffer_load_i64_inline(buffer, offset, result);
    });

    // TODO(benvanik): rework dispatch so that the StoreI* ops can share the
    // same body - they only vary on the length.
    // See VMOpcodesCore.td for more information on the encoding.
    DISPATCH_OP(CORE, BufferStoreI8, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "target_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      uint8_t value = (uint8_t)VM_DecOperandRegI32("value");
      vm_buffer_store_i8_inline(buffer, offset, value);
    });
    DISPATCH_OP(CORE, BufferStoreI16, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "target_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      uint16_t value = (uint16_t)VM_DecOperandRegI32("value");
      vm_buffer_store_i16_inline(buffer, offset, value);
    });
    DISPATCH_OP(CORE, BufferStoreI32, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "target_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      uint32_t value = VM_DecOperandRegI32("value");
      vm_buffer_store_i32_inline(buffer, offset, value);
    });
    DISPATCH_OP(CORE, BufferStoreI64, {
      bool buffer_is_move;
      iree_vm_ref_t* buffer_ref =
          VM_DecOperandRegRef("target_buffer", &buffer_is_move);
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
      if (IREE_UNLIKELY(!buffer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "target_buffer is null");
      }
      iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
      uint64_t value = (uint64_t)VM_DecOperandRegI64("value");
      vm_buffer_store_i64_inline(buffer, offset, value);
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
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t minimum_capacity = VM_DecOperandRegI32("minimum_capacity");
      IREE_RETURN_IF_ERROR(iree_vm_list_reserve(list, minimum_capacity));
    });

    DISPATCH_OP(CORE, ListSize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      int32_t* result = VM_DecResultRegI32("result");
      *result = (int32_t)iree_vm_list_size(list);
    });

    DISPATCH_OP(CORE, ListResize, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t new_size = VM_DecOperandRegI32("new_size");
      IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, new_size));
    });

    DISPATCH_OP(CORE, ListGetI32, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
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
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t index = VM_DecOperandRegI32("index");
      int32_t raw_value = VM_DecOperandRegI32("raw_value");
      iree_vm_value_t value = iree_vm_value_make_i32(raw_value);
      IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
    });

    DISPATCH_OP(CORE, ListGetI64, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
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

    DISPATCH_OP(CORE, ListSetI64, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t index = VM_DecOperandRegI32("index");
      int64_t raw_value = VM_DecOperandRegI64("value");
      iree_vm_value_t value = iree_vm_value_make_i64(raw_value);
      IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
    });

    DISPATCH_OP(CORE, ListGetRef, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t index = VM_DecOperandRegI32("index");
      const iree_vm_type_def_t* type_def = VM_DecTypeOf("result");
      bool result_is_move;
      iree_vm_ref_t* result = VM_DecResultRegRef("result", &result_is_move);
      // TODO(benvanik): use result_is_move with a _retain_or_move.
      IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_retain(list, index, result));
      if (result->type != IREE_VM_REF_TYPE_NULL &&
          (iree_vm_type_def_is_value(type_def) ||
           result->type != type_def->ref_type)) {
        // Type mismatch; put null in the register instead.
        // TODO(benvanik): return an error here and make a query type method?
        iree_vm_ref_release(result);
      }
    });

    DISPATCH_OP(CORE, ListSetRef, {
      bool list_is_move;
      iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
      iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
      if (IREE_UNLIKELY(!list)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
      }
      uint32_t index = VM_DecOperandRegI32("index");
      bool operand_is_move;
      iree_vm_ref_t* operand = VM_DecOperandRegRef("value", &operand_is_move);
      if (operand_is_move) {
        IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_move(list, index, operand));
      } else {
        IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_retain(list, index, operand));
      }
    });

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, SelectI32, {
      int32_t condition = VM_DecOperandRegI32("condition");
      int32_t true_value = VM_DecOperandRegI32("true_value");
      int32_t false_value = VM_DecOperandRegI32("false_value");
      int32_t* result = VM_DecResultRegI32("result");
      *result = vm_select_i32(condition, true_value, false_value);
    });

    DISPATCH_OP(CORE, SelectI64, {
      int32_t condition = VM_DecOperandRegI32("condition");
      int64_t true_value = VM_DecOperandRegI64("true_value");
      int64_t false_value = VM_DecOperandRegI64("false_value");
      int64_t* result = VM_DecResultRegI64("result");
      *result = vm_select_i64(condition, true_value, false_value);
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
        if (false_value_is_move && false_value != result) {
          iree_vm_ref_release(false_value);
        }
      } else {
        // Select RHS.
        IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
            false_value_is_move, false_value, type_def->ref_type, result));
        if (true_value_is_move && true_value != result) {
          iree_vm_ref_release(true_value);
        }
      }
    });

    DISPATCH_OP(CORE, SwitchI32, {
      int32_t index = VM_DecOperandRegI32("index");
      int32_t default_value = VM_DecIntAttr32("default_value");
      const iree_vm_register_list_t* value_reg_list =
          VM_DecVariadicOperands("values");
      int32_t* result = VM_DecResultRegI32("result");
      if (index >= 0 && index < value_reg_list->size) {
        *result = regs_i32[value_reg_list->registers[index]];
      } else {
        *result = default_value;
      }
    });

    DISPATCH_OP(CORE, SwitchI64, {
      int32_t index = VM_DecOperandRegI32("index");
      int64_t default_value = VM_DecIntAttr64("default_value");
      const iree_vm_register_list_t* value_reg_list =
          VM_DecVariadicOperands("values");
      int64_t* result = VM_DecResultRegI64("result");
      if (index >= 0 && index < value_reg_list->size) {
        *result = regs_i32[value_reg_list->registers[index]];
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
        iree_vm_ref_t* new_value = &regs_ref[value_reg_list->registers[index] &
                                             IREE_REF_REGISTER_MASK];
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

    DISPATCH_OP_CORE_BINARY_I32(AddI32, vm_add_i32);
    DISPATCH_OP_CORE_BINARY_I32(SubI32, vm_sub_i32);
    DISPATCH_OP_CORE_BINARY_I32(MulI32, vm_mul_i32);
    DISPATCH_OP_CORE_BINARY_I32(DivI32S, vm_div_i32s);
    DISPATCH_OP_CORE_BINARY_I32(DivI32U, vm_div_i32u);
    DISPATCH_OP_CORE_BINARY_I32(RemI32S, vm_rem_i32s);
    DISPATCH_OP_CORE_BINARY_I32(RemI32U, vm_rem_i32u);
    DISPATCH_OP_CORE_TERNARY_I32(FMAI32, vm_fma_i32);
    DISPATCH_OP_CORE_UNARY_I32(AbsI32, vm_abs_i32);
    DISPATCH_OP_CORE_BINARY_I32(MinI32S, vm_min_i32s);
    DISPATCH_OP_CORE_BINARY_I32(MinI32U, vm_min_i32u);
    DISPATCH_OP_CORE_BINARY_I32(MaxI32S, vm_max_i32s);
    DISPATCH_OP_CORE_BINARY_I32(MaxI32U, vm_max_i32u);
    DISPATCH_OP_CORE_UNARY_I32(NotI32, vm_not_i32);
    DISPATCH_OP_CORE_BINARY_I32(AndI32, vm_and_i32);
    DISPATCH_OP_CORE_BINARY_I32(OrI32, vm_or_i32);
    DISPATCH_OP_CORE_BINARY_I32(XorI32, vm_xor_i32);
    DISPATCH_OP_CORE_UNARY_I32(CtlzI32, vm_ctlz_i32);

    DISPATCH_OP_CORE_BINARY_I64(AddI64, vm_add_i64);
    DISPATCH_OP_CORE_BINARY_I64(SubI64, vm_sub_i64);
    DISPATCH_OP_CORE_BINARY_I64(MulI64, vm_mul_i64);
    DISPATCH_OP_CORE_BINARY_I64(DivI64S, vm_div_i64s);
    DISPATCH_OP_CORE_BINARY_I64(DivI64U, vm_div_i64u);
    DISPATCH_OP_CORE_BINARY_I64(RemI64S, vm_rem_i64s);
    DISPATCH_OP_CORE_BINARY_I64(RemI64U, vm_rem_i64u);
    DISPATCH_OP_CORE_TERNARY_I64(FMAI64, vm_fma_i64);
    DISPATCH_OP_CORE_UNARY_I64(AbsI64, vm_abs_i64);
    DISPATCH_OP_CORE_BINARY_I64(MinI64S, vm_min_i64s);
    DISPATCH_OP_CORE_BINARY_I64(MinI64U, vm_min_i64u);
    DISPATCH_OP_CORE_BINARY_I64(MaxI64S, vm_max_i64s);
    DISPATCH_OP_CORE_BINARY_I64(MaxI64U, vm_max_i64u);
    DISPATCH_OP_CORE_UNARY_I64(NotI64, vm_not_i64);
    DISPATCH_OP_CORE_BINARY_I64(AndI64, vm_and_i64);
    DISPATCH_OP_CORE_BINARY_I64(OrI64, vm_or_i64);
    DISPATCH_OP_CORE_BINARY_I64(XorI64, vm_xor_i64);
    DISPATCH_OP_CORE_UNARY_I64(CtlzI64, vm_ctlz_i64);

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

    // NOTE: these all operate on 32-bit registers.
    DISPATCH_OP_CORE_UNARY_I32(TruncI32I8, vm_trunc_i32i8);
    DISPATCH_OP_CORE_UNARY_I32(TruncI32I16, vm_trunc_i32i16);
    DISPATCH_OP_CORE_UNARY_I32(ExtI8I32S, vm_ext_i8i32s);
    DISPATCH_OP_CORE_UNARY_I32(ExtI8I32U, vm_ext_i8i32u);
    DISPATCH_OP_CORE_UNARY_I32(ExtI16I32S, vm_ext_i16i32s);
    DISPATCH_OP_CORE_UNARY_I32(ExtI16I32U, vm_ext_i16i32u);

    // NOTE: 64-bit ones are actually changing register widths.
    DISPATCH_OP(CORE, TruncI64I32, {
      int64_t operand = VM_DecOperandRegI64("operand");
      int32_t* result = VM_DecResultRegI32("result");
      *result = vm_trunc_i64i32(operand);
    });
    DISPATCH_OP(CORE, ExtI32I64S, {
      int32_t operand = VM_DecOperandRegI32("operand");
      int64_t* result = VM_DecResultRegI64("result");
      *result = vm_ext_i32i64s(operand);
    });
    DISPATCH_OP(CORE, ExtI32I64U, {
      int32_t operand = VM_DecOperandRegI32("operand");
      int64_t* result = VM_DecResultRegI64("result");
      *result = vm_ext_i32i64u(operand);
    });

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define DISPATCH_OP_CORE_SHIFT_I32(op_name, op_func)  \
  DISPATCH_OP(CORE, op_name, {                        \
    int32_t operand = VM_DecOperandRegI32("operand"); \
    int32_t amount = VM_DecOperandRegI32("amount");   \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = op_func(operand, amount);               \
  });

    DISPATCH_OP_CORE_SHIFT_I32(ShlI32, vm_shl_i32);
    DISPATCH_OP_CORE_SHIFT_I32(ShrI32S, vm_shr_i32s);
    DISPATCH_OP_CORE_SHIFT_I32(ShrI32U, vm_shr_i32u);

#define DISPATCH_OP_CORE_SHIFT_I64(op_name, op_func)  \
  DISPATCH_OP(CORE, op_name, {                        \
    int64_t operand = VM_DecOperandRegI64("operand"); \
    int32_t amount = VM_DecOperandRegI32("amount");   \
    int64_t* result = VM_DecResultRegI64("result");   \
    *result = op_func(operand, amount);               \
  });

    DISPATCH_OP_CORE_SHIFT_I64(ShlI64, vm_shl_i64);
    DISPATCH_OP_CORE_SHIFT_I64(ShrI64S, vm_shr_i64s);
    DISPATCH_OP_CORE_SHIFT_I64(ShrI64U, vm_shr_i64u);

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

    DISPATCH_OP_CORE_BINARY_I32(CmpEQI32, vm_cmp_eq_i32);
    DISPATCH_OP_CORE_BINARY_I32(CmpNEI32, vm_cmp_ne_i32);
    DISPATCH_OP_CORE_BINARY_I32(CmpLTI32S, vm_cmp_lt_i32s);
    DISPATCH_OP_CORE_BINARY_I32(CmpLTI32U, vm_cmp_lt_i32u);
    DISPATCH_OP_CORE_UNARY_I32(CmpNZI32, vm_cmp_nz_i32);

#define DISPATCH_OP_CORE_CMP_I64(op_name, op_func)  \
  DISPATCH_OP(CORE, op_name, {                      \
    int64_t lhs = VM_DecOperandRegI64("lhs");       \
    int64_t rhs = VM_DecOperandRegI64("rhs");       \
    int32_t* result = VM_DecResultRegI32("result"); \
    *result = op_func(lhs, rhs);                    \
  });

    DISPATCH_OP_CORE_CMP_I64(CmpEQI64, vm_cmp_eq_i64);
    DISPATCH_OP_CORE_CMP_I64(CmpNEI64, vm_cmp_ne_i64);
    DISPATCH_OP_CORE_CMP_I64(CmpLTI64S, vm_cmp_lt_i64s);
    DISPATCH_OP_CORE_CMP_I64(CmpLTI64U, vm_cmp_lt_i64u);
    DISPATCH_OP(CORE, CmpNZI64, {
      int64_t operand = VM_DecOperandRegI64("operand");
      int32_t* result = VM_DecResultRegI32("result");
      *result = vm_cmp_nz_i64(operand);
    });

    DISPATCH_OP(CORE, CmpEQRef, {
      bool lhs_is_move;
      iree_vm_ref_t* lhs = VM_DecOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      iree_vm_ref_t* rhs = VM_DecOperandRegRef("rhs", &rhs_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = vm_cmp_eq_ref(lhs, rhs);
      if (lhs_is_move) iree_vm_ref_release(lhs);
      if (rhs_is_move) iree_vm_ref_release(rhs);
    });
    DISPATCH_OP(CORE, CmpNERef, {
      bool lhs_is_move;
      iree_vm_ref_t* lhs = VM_DecOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      iree_vm_ref_t* rhs = VM_DecOperandRegRef("rhs", &rhs_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = vm_cmp_ne_ref(lhs, rhs);
      if (lhs_is_move) iree_vm_ref_release(lhs);
      if (rhs_is_move) iree_vm_ref_release(rhs);
    });
    DISPATCH_OP(CORE, CmpNZRef, {
      bool operand_is_move;
      iree_vm_ref_t* operand = VM_DecOperandRegRef("operand", &operand_is_move);
      int32_t* result = VM_DecResultRegI32("result");
      *result = vm_cmp_nz_ref(operand);
      if (operand_is_move) iree_vm_ref_release(operand);
    });

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    // No-op in the interpreter.
    DISPATCH_OP(CORE, Block, {});

    DISPATCH_OP(CORE, Branch, {
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      pc = block_pc + IREE_VM_BLOCK_MARKER_SIZE;  // skip block marker
      if (IREE_UNLIKELY(remap_list->size > 0)) {
        iree_vm_bytecode_dispatch_remap_branch_registers(regs_i32, regs_ref,
                                                         remap_list);
      }
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
        pc = true_block_pc + IREE_VM_BLOCK_MARKER_SIZE;  // skip block marker
        if (IREE_UNLIKELY(true_remap_list->size > 0)) {
          iree_vm_bytecode_dispatch_remap_branch_registers(regs_i32, regs_ref,
                                                           true_remap_list);
        }
      } else {
        pc = false_block_pc + IREE_VM_BLOCK_MARKER_SIZE;  // skip block marker
        if (IREE_UNLIKELY(false_remap_list->size > 0)) {
          iree_vm_bytecode_dispatch_remap_branch_registers(regs_i32, regs_ref,
                                                           false_remap_list);
        }
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
            dst_reg_list, &current_frame, &regs));
      } else {
        // Switch execution to the target function and continue running in the
        // bytecode dispatcher.
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_internal_enter(
            stack, current_frame->function.module, function_ordinal,
            src_reg_list, dst_reg_list, &current_frame, &regs));
        bytecode_data =
            module->bytecode_data.data +
            module->function_descriptor_table[function_ordinal].bytecode_offset;
        regs_i32 = regs.i32;
        IREE_BUILTIN_ASSUME_ALIGNED(regs_i32, 16);
        regs_ref = regs.ref;
        IREE_BUILTIN_ASSUME_ALIGNED(regs_ref, 16);
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
          src_reg_list, dst_reg_list, &current_frame, &regs));
    });

    DISPATCH_OP(CORE, Return, {
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      current_frame->pc = pc;

      // TODO(benvanik): faster check for escaping; this is slow (cache misses).
      iree_vm_stack_frame_t* parent_frame = iree_vm_stack_parent_frame(stack);
      if (!parent_frame ||
          parent_frame->module_state != current_frame->module_state) {
        // Return from the top-level entry frame - return back to call().
        return iree_vm_bytecode_external_leave(stack, current_frame, &regs,
                                               src_reg_list, call_results);
      }

      // Store results into the caller frame and pop back to the parent.
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_internal_leave(
          stack, current_frame, regs, src_reg_list, &current_frame, &regs));

      // Reset dispatch state so we can continue executing in the caller.
      bytecode_data =
          module->bytecode_data.data +
          module->function_descriptor_table[current_frame->function.ordinal]
              .bytecode_offset;
      regs_i32 = regs.i32;
      IREE_BUILTIN_ASSUME_ALIGNED(regs_i32, 16);
      regs_ref = regs.ref;
      IREE_BUILTIN_ASSUME_ALIGNED(regs_ref, 16);
      pc = current_frame->pc;
    });

    DISPATCH_OP(CORE, Fail, {
      uint32_t status_code = VM_DecOperandRegI32("status");
      iree_string_view_t message;
      VM_DecStrAttr("message", &message);
      if (status_code != 0) {
        // TODO(benvanik): capture source information.
        return iree_status_allocate_f(status_code, "<vm>", 0, "%.*s",
                                      (int)message.size, message.data);
      }
    });

    DISPATCH_OP(CORE, ImportResolved, {
      uint32_t function_ordinal = VM_DecFuncAttr("import");
      int32_t* result = VM_DecResultRegI32("result");
      uint32_t import_ordinal = function_ordinal & 0x7FFFFFFFu;
      IREE_ASSERT(import_ordinal < module_state->import_count);
      const iree_vm_bytecode_import_t* import =
          &module_state->import_table[import_ordinal];
      *result = import->function.module != NULL ? 1 : 0;
    });

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    DISPATCH_OP(CORE, Yield, {
      // Perform branch before yielding; in this way we will resume at the
      // target without needing to retain any information about the yield.
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      iree_vm_bytecode_dispatch_remap_branch_registers(regs_i32, regs_ref,
                                                       remap_list);
      current_frame->pc =
          block_pc + IREE_VM_BLOCK_MARKER_SIZE;  // skip block marker

      // Return magic status code indicating a yield.
      // This isn't an error, though callers not supporting coroutines will
      // treat it as one and propagate it up.
      return iree_status_from_code(IREE_STATUS_DEFERRED);
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
      iree_vm_bytecode_dispatch_discard_registers(regs_ref, src_reg_list);
    });

    DISPATCH_OP(CORE, Print, {
      iree_string_view_t event_name;
      VM_DecStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_DecVariadicOperands("operands");
      // TODO(benvanik): print.
      iree_vm_bytecode_dispatch_discard_registers(regs_ref, src_reg_list);
    });

    DISPATCH_OP(CORE, Break, {
      // TODO(benvanik): break unconditionally.
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_DecBranchOperands("operands");
      iree_vm_bytecode_dispatch_remap_branch_registers(regs_i32, regs_ref,
                                                       remap_list);
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
      iree_vm_bytecode_dispatch_remap_branch_registers(regs_i32, regs_ref,
                                                       remap_list);
      pc = block_pc + IREE_VM_BLOCK_MARKER_SIZE;  // skip block marker
    });

    //===------------------------------------------------------------------===//
    // Extension trampolines
    //===------------------------------------------------------------------===//

#if IREE_VM_EXT_F32_ENABLE
    BEGIN_DISPATCH_PREFIX(PrefixExtF32, EXT_F32) {
      //===----------------------------------------------------------------===//
      // ExtF32: Globals
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_F32, GlobalLoadF32, {
        uint32_t byte_offset = VM_DecGlobalAttr("global");
        IREE_ASSERT(byte_offset + 4 <=
                    module_state->rwdata_storage.data_length);
        float* value = VM_DecResultRegF32("value");
        const float global_value =
            vm_global_load_f32(module_state->rwdata_storage.data, byte_offset);
        *value = global_value;
      });

      DISPATCH_OP(EXT_F32, GlobalStoreF32, {
        uint32_t byte_offset = VM_DecGlobalAttr("global");
        IREE_ASSERT(byte_offset + 4 <=
                    module_state->rwdata_storage.data_length);
        float value = VM_DecOperandRegF32("value");
        vm_global_store_f32(module_state->rwdata_storage.data, byte_offset,
                            value);
      });

      DISPATCH_OP(EXT_F32, GlobalLoadIndirectF32, {
        uint32_t byte_offset = VM_DecOperandRegI32("global");
        if (IREE_UNLIKELY(byte_offset + 4 >
                          module_state->rwdata_storage.data_length)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
              module_state->rwdata_storage.data_length);
        }
        float* value = VM_DecResultRegF32("value");
        const float global_value =
            vm_global_load_f32(module_state->rwdata_storage.data, byte_offset);
        *value = global_value;
      });

      DISPATCH_OP(EXT_F32, GlobalStoreIndirectF32, {
        uint32_t byte_offset = VM_DecOperandRegI32("global");
        if (IREE_UNLIKELY(byte_offset + 4 >
                          module_state->rwdata_storage.data_length)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "global byte_offset out of range: %d (rwdata=%zu)", byte_offset,
              module_state->rwdata_storage.data_length);
        }
        float value = VM_DecOperandRegF32("value");
        vm_global_store_f32(module_state->rwdata_storage.data, byte_offset,
                            value);
      });

      //===----------------------------------------------------------------===//
      // ExtF32: Constants
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_F32, ConstF32, {
        float value = VM_DecFloatAttr32("value");
        float* result = VM_DecResultRegF32("result");
        *result = value;
      });

      DISPATCH_OP(EXT_F32, ConstF32Zero, {
        float* result = VM_DecResultRegF32("result");
        *result = 0;
      });

      //===----------------------------------------------------------------===//
      // ExtF32: Lists
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_F32, ListGetF32, {
        bool list_is_move;
        iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
        iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
        if (IREE_UNLIKELY(!list)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
        }
        uint32_t index = VM_DecOperandRegI32("index");
        float* result = VM_DecResultRegF32("result");
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            list, index, IREE_VM_VALUE_TYPE_F32, &value));
        *result = value.f32;
      });

      DISPATCH_OP(EXT_F32, ListSetF32, {
        bool list_is_move;
        iree_vm_ref_t* list_ref = VM_DecOperandRegRef("list", &list_is_move);
        iree_vm_list_t* list = iree_vm_list_deref(*list_ref);
        if (IREE_UNLIKELY(!list)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "list is null");
        }
        uint32_t index = VM_DecOperandRegI32("index");
        float raw_value = VM_DecOperandRegF32("value");
        iree_vm_value_t value = iree_vm_value_make_f32(raw_value);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(list, index, &value));
      });

      //===----------------------------------------------------------------===//
      // ExtF32: Conditional assignment
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_F32, SelectF32, {
        int32_t condition = VM_DecOperandRegI32("condition");
        float true_value = VM_DecOperandRegF32("true_value");
        float false_value = VM_DecOperandRegF32("false_value");
        float* result = VM_DecResultRegF32("result");
        *result = vm_select_f32(condition, true_value, false_value);
      });

      DISPATCH_OP(EXT_F32, SwitchF32, {
        int32_t index = VM_DecOperandRegI32("index");
        float default_value = VM_DecFloatAttr32("default_value");
        const iree_vm_register_list_t* value_reg_list =
            VM_DecVariadicOperands("values");
        float* result = VM_DecResultRegF32("result");
        if (index >= 0 && index < value_reg_list->size) {
          *result = *((float*)&regs_i32[value_reg_list->registers[index]]);
        } else {
          *result = default_value;
        }
      });

      //===----------------------------------------------------------------===//
      // ExtF32: Native floating-point arithmetic
      //===----------------------------------------------------------------===//

      DISPATCH_OP_EXT_F32_BINARY_F32(AddF32, vm_add_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(SubF32, vm_sub_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(MulF32, vm_mul_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(DivF32, vm_div_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(RemF32, vm_rem_f32);
      DISPATCH_OP_EXT_F32_TERNARY_F32(FMAF32, vm_fma_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(AbsF32, vm_abs_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(NegF32, vm_neg_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(CeilF32, vm_ceil_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(FloorF32, vm_floor_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(RoundF32, vm_round_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(MinF32, vm_min_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(MaxF32, vm_max_f32);

      DISPATCH_OP_EXT_F32_UNARY_F32(AtanF32, vm_atan_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(Atan2F32, vm_atan2_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(CosF32, vm_cos_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(SinF32, vm_sin_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(ExpF32, vm_exp_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(Exp2F32, vm_exp2_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(ExpM1F32, vm_expm1_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(LogF32, vm_log_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(Log10F32, vm_log10_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(Log1pF32, vm_log1p_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(Log2F32, vm_log2_f32);
      DISPATCH_OP_EXT_F32_BINARY_F32(PowF32, vm_pow_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(RsqrtF32, vm_rsqrt_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(SqrtF32, vm_sqrt_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(TanhF32, vm_tanh_f32);
      DISPATCH_OP_EXT_F32_UNARY_F32(ErfF32, vm_erf_f32);

      //===----------------------------------------------------------------===//
      // ExtF32: Casting and type conversion/emulation
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_F32, CastSI32F32, {
        int32_t operand = (int32_t)VM_DecOperandRegI32("operand");
        float* result = VM_DecResultRegF32("result");
        *result = vm_cast_si32f32(operand);
      });
      DISPATCH_OP(EXT_F32, CastUI32F32, {
        int32_t operand = (int32_t)VM_DecOperandRegI32("operand");
        float* result = VM_DecResultRegF32("result");
        *result = vm_cast_ui32f32(operand);
      });
      DISPATCH_OP(EXT_F32, CastF32SI32, {
        float operand = VM_DecOperandRegF32("operand");
        int32_t* result = VM_DecResultRegI32("result");
        *result = vm_cast_f32si32(operand);
      });
      DISPATCH_OP(EXT_F32, CastF32UI32, {
        float operand = VM_DecOperandRegF32("operand");
        int32_t* result = VM_DecResultRegI32("result");
        *result = vm_cast_f32ui32(operand);
      });
      DISPATCH_OP(EXT_F32, BitcastI32F32, {
        int32_t operand = (int32_t)VM_DecOperandRegI32("operand");
        float* result = VM_DecResultRegF32("result");
        *result = vm_bitcast_i32f32(operand);
      });
      DISPATCH_OP(EXT_F32, BitcastF32I32, {
        float operand = VM_DecOperandRegF32("operand");
        int32_t* result = VM_DecResultRegI32("result");
        *result = vm_bitcast_f32i32(operand);
      });

      //===----------------------------------------------------------------===//
      // ExtF32: Comparison ops
      //===----------------------------------------------------------------===//

#define DISPATCH_OP_EXT_F32_CMP_F32(op_name, op_func) \
  DISPATCH_OP(EXT_F32, op_name, {                     \
    float lhs = VM_DecOperandRegF32("lhs");           \
    float rhs = VM_DecOperandRegF32("rhs");           \
    int32_t* result = VM_DecResultRegI32("result");   \
    *result = op_func(lhs, rhs);                      \
  });

      DISPATCH_OP_EXT_F32_CMP_F32(CmpEQF32O, vm_cmp_eq_f32o);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpEQF32U, vm_cmp_eq_f32u);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpNEF32O, vm_cmp_ne_f32o);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpNEF32U, vm_cmp_ne_f32u);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpLTF32O, vm_cmp_lt_f32o);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpLTF32U, vm_cmp_lt_f32u);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpLTEF32O, vm_cmp_lte_f32o);
      DISPATCH_OP_EXT_F32_CMP_F32(CmpLTEF32U, vm_cmp_lte_f32u);
      DISPATCH_OP(EXT_F32, CmpNaNF32, {
        float operand = VM_DecOperandRegF32("operand");
        int32_t* result = VM_DecResultRegI32("result");
        *result = vm_cmp_nan_f32(operand);
      });

      //===----------------------------------------------------------------===//
      // ExtF32: Buffers
      //===----------------------------------------------------------------===//

      DISPATCH_OP(EXT_F32, BufferFillF32, {
        bool buffer_is_move;
        iree_vm_ref_t* buffer_ref =
            VM_DecOperandRegRef("target_buffer", &buffer_is_move);
        iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
        if (IREE_UNLIKELY(!buffer)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "buffer is null");
        }
        iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
        iree_host_size_t length = VM_DecOperandRegI64HostSize("length");
        float value = VM_DecOperandRegF32("value");
        IREE_RETURN_IF_ERROR(vm_buffer_fill_f32(buffer, offset, length, value));
      });

      DISPATCH_OP(EXT_F32, BufferLoadF32, {
        bool buffer_is_move;
        iree_vm_ref_t* buffer_ref =
            VM_DecOperandRegRef("source_buffer", &buffer_is_move);
        iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
        if (IREE_UNLIKELY(!buffer)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source_buffer is null");
        }
        iree_host_size_t offset = VM_DecOperandRegI64HostSize("source_offset");
        float* result = VM_DecResultRegF32("result");
        vm_buffer_load_f32_inline(buffer, offset, result);
      });

      DISPATCH_OP(EXT_F32, BufferStoreF32, {
        bool buffer_is_move;
        iree_vm_ref_t* buffer_ref =
            VM_DecOperandRegRef("target_buffer", &buffer_is_move);
        iree_vm_buffer_t* buffer = iree_vm_buffer_deref(*buffer_ref);
        if (IREE_UNLIKELY(!buffer)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "target_buffer is null");
        }
        iree_host_size_t offset = VM_DecOperandRegI64HostSize("target_offset");
        float value = VM_DecOperandRegF32("value");
        vm_buffer_store_f32_inline(buffer, offset, value);
      });
    }
    END_DISPATCH_PREFIX();
#else
    UNHANDLED_DISPATCH_PREFIX(PrefixExtF32, EXT_F32);
#endif  // IREE_VM_EXT_F32_ENABLE

    DISPATCH_OP(CORE, PrefixExtF64,
                { return iree_make_status(IREE_STATUS_UNIMPLEMENTED); });

    // NOLINTNEXTLINE(misc-static-assert)
    DISPATCH_UNHANDLED_CORE();
  }
  END_DISPATCH_CORE();
}
