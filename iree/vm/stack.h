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

#ifndef IREE_VM_STACK_H_
#define IREE_VM_STACK_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A reasonable default stack storage size, in bytes.
// This will allow most (reasonable) programs to run. If running
// unverified/untested programs then prefer to use a dynamically growable stack
// until the expectations of the programs are checked; for example, hopefully
// in a year or two we have much more complex models with much deeper call
// stacks and we may want to re-evaluate the host-stack allocation size.
//
// The value was chosen to fit quite a few i32 registers and a reasonable amount
// of ref registers (that are 2 * sizeof(void*)). For many invocations this will
// be more than enough to perform the work without needing an additional dynamic
// allocation/resize.
#define IREE_VM_STACK_DEFAULT_SIZE (8 * 1024)

// The minimum size of VM stack storage.
#define IREE_VM_STACK_MIN_SIZE (1 * 1024)

// The maximum size of VM stack storage; anything larger is probably a bug.
#define IREE_VM_STACK_MAX_SIZE (1 * 1024 * 1024)

// Maximum register count per bank.
// This determines the bits required to reference registers in the VM bytecode.
#define IREE_I32_REGISTER_COUNT 0x7FFF
#define IREE_REF_REGISTER_COUNT 0x7FFF

#define IREE_I32_REGISTER_MASK 0x7FFF

#define IREE_REF_REGISTER_TYPE_BIT 0x8000
#define IREE_REF_REGISTER_MOVE_BIT 0x4000
#define IREE_REF_REGISTER_MASK 0x3FFF

// Pointers to typed register storage.
typedef struct {
  // Ordinal mask defining which ordinal bits are valid. All i32 indexing must
  // be ANDed with this mask.
  uint16_t i32_mask;
  // Ordinal mask defining which ordinal bits are valid. All ref indexing must
  // be ANDed with this mask.
  uint16_t ref_mask;
  // 16-byte aligned i32 register array.
  int32_t* i32;
  // Naturally aligned ref register array.
  iree_vm_ref_t* ref;
} iree_vm_registers_t;

// A single stack frame within the VM.
typedef struct iree_vm_stack_frame {
  // NOTE: to (try to) get better cache hit rates we put the most frequently
  // accessed members first.

  // Current program counter within the function.
  // Implementations may treat this offset differently, treating it as a byte
  // offset (such as in the case of VM bytecode), a block identifier (compiled
  // code), etc.
  iree_vm_source_offset_t pc;

  // Pointers to register arrays into storage.
  iree_vm_registers_t registers;

  // Function that the stack frame is within.
  iree_vm_function_t function;

  // Cached module state pointer for the module containing |function|.
  // This removes the need to lookup the module state when control returns to
  // the function during continuation or from a return instruction.
  iree_vm_module_state_t* module_state;

  // Depth of the frame within the stack.
  // As stack frame pointers are not stable this can be used instead to detect
  // stack enter/leave balance issues.
  int32_t depth;
} iree_vm_stack_frame_t;

// A state resolver that can allocate or lookup module state.
typedef struct iree_vm_state_resolver {
  void* self;
  iree_status_t(IREE_API_PTR* query_module_state)(
      void* state_resolver, iree_vm_module_t* module,
      iree_vm_module_state_t** out_module_state);
} iree_vm_state_resolver_t;

// A fiber stack used for storing stack frame state during execution.
// All required state is stored within the stack and no host thread-local state
// is used allowing us to execute multiple fibers on the same host thread.
typedef struct iree_vm_stack iree_vm_stack_t;

// Defines and initializes an inline VM stack.
// The stack will be ready for use and must be deinitialized with
// iree_vm_stack_deinitialize when no longer required.
//
// Example:
//  IREE_VM_INLINE_STACK_INITIALIZE(
//      stack,
//      iree_vm_context_state_resolver(context),
//      iree_allocator_system());
//  ...
//  iree_vm_stack_deinitialize(stack);
#define IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, allocator) \
  uint8_t __stack_storage[IREE_VM_STACK_DEFAULT_SIZE];                    \
  iree_byte_span_t __stack_storage_span =                                 \
      iree_make_byte_span(__stack_storage, sizeof(__stack_storage));      \
  iree_vm_stack_t* stack = NULL;                                          \
  IREE_IGNORE_ERROR(iree_vm_stack_initialize(                             \
      __stack_storage_span, (state_resolver), (allocator), &stack));

// Initializes a statically-allocated stack in |storage|.
// The contents of the |storage| can be anything upon initialization and the
// stack must be deinitialized with iree_vm_stack_deinitialize before the
// storage is freed. The provided |allocator| is only used for stack growth
// beyond the intial storage capacity and may be iree_allocator_null() to
// prevent growth. Use IREE_VM_STACK_DEFAULT_SIZE for a reasonable default or
// use iree_vm_stack_allocate if the input programs may exceed reason.
//
// The provided |state_resolver| will be used to resolve a module to a module
// state within a context. This will be called on function entry whenever module
// transitions occur.
//
// Example:
//  uint8_t stack_storage[IREE_VM_STACK_DEFAULT_SIZE];
//  iree_vm_stack_t* stack = NULL;
//  iree_vm_stack_initialize(stack_storage, ..., &stack);
//  ...
//  iree_vm_stack_deinitialize(stack);
//  // stack_storage can now be reused/freed/etc
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_initialize(
    iree_byte_span_t storage, iree_vm_state_resolver_t state_resolver,
    iree_allocator_t allocator, iree_vm_stack_t** out_stack);

// Deinitializes a statically-allocated |stack| previously initialized with
// iree_vm_stack_initialize.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_stack_deinitialize(iree_vm_stack_t* stack);

// Allocates a dynamically-growable stack.
//
// The provided |state_resolver| will be used to resolve a module to a module
// state within a context. This will be called on function entry whenever module
// transitions occur.
//
// The stack will be allocated from |allocator| and returned in |out_stack|.
// It must be freed with iree_vm_stack_free.
//
// Example:
//  iree_vm_stack_t* stack = NULL;
//  iree_vm_stack_allocate(..., iree_allocator_system(), &stack);
//  ...
//  iree_vm_stack_free(stack);
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_allocate(iree_vm_state_resolver_t state_resolver,
                       iree_allocator_t allocator, iree_vm_stack_t** out_stack);

// Frees a dynamically-allocated |stack| from iree_vm_stack_allocate.
IREE_API_EXPORT void IREE_API_CALL iree_vm_stack_free(iree_vm_stack_t* stack);

// Returns the current stack frame or nullptr if the stack is empty.
IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_current_frame(iree_vm_stack_t* stack);

// Returns the parent stack frame or nullptr if the stack is empty.
IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_parent_frame(iree_vm_stack_t* stack);

// Queries the context-specific module state for the given module.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_query_module_state(
    iree_vm_stack_t* stack, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state);

// Enters into the given |function| and returns the callee stack frame.
// May invalidate any pointers to stack frames and the only pointer that can be
// assumed valid after return is the one in |out_callee_frame|.
//
// |argument_registers| and |result_registers| are lists of registers within the
// caller frame that will be used to source and store arguments and results with
// the callee. They must remain live and valid until the callee returns (which
// may be much later if asynchronous!).
//
// TODO(benvanik): copy the result register list to make async easier.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_function_enter(
    iree_vm_stack_t* stack, iree_vm_function_t function,
    const iree_vm_register_list_t* argument_registers,
    const iree_vm_register_list_t* result_registers,
    iree_vm_stack_frame_t** out_callee_frame);

// Leaves the current stack frame.
// The provided |result_registers| in the callee frame will be stored back into
// the caller frame result registers.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_function_leave(
    iree_vm_stack_t* stack, const iree_vm_register_list_t* result_registers,
    iree_vm_stack_frame_t** out_caller_frame);

// Enters into an `[external]` frame and returns the external stack frame.
// May invalidate any existing pointers to stack frames and the only pointer
// that can be assumed valid after return is the one in |out_callee_frame|.
//
// The frame will have |register_count| registers of each type reserved and
// ready for the caller to populate.
//
// |name| can be used to provide a debug-visible name to help identify where the
// transition is occurring.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_external_enter(
    iree_vm_stack_t* stack, iree_string_view_t name,
    iree_host_size_t register_count, iree_vm_stack_frame_t** out_callee_frame);

// Leaves the current external stack frame.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_external_leave(iree_vm_stack_t* stack);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_STACK_H_
