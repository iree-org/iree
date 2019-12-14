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

#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm2/module.h"
#include "iree/vm2/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum stack depth, in frames.
#define IREE_MAX_STACK_DEPTH 32

// Maximum register count per bank.
// This determines the bits required to reference registers in the VM bytecode.
#define IREE_I32_REGISTER_COUNT 128
#define IREE_REF_REGISTER_COUNT 64

#define IREE_I32_REGISTER_MASK 0x7F

#define IREE_REF_REGISTER_TYPE_BIT 0x80
#define IREE_REF_REGISTER_MOVE_BIT 0x40
#define IREE_REF_REGISTER_MASK 0x3F

// An opaque offset into a source map that a source resolver can calculate.
// Do not assume that iree_vm_source_offset_t+1 means the next byte offset as
// backends are free to treat these as everything from pointers to machine code
// to hash codes.
typedef int64_t iree_vm_source_offset_t;

// Register banks for use within a stack frame.
typedef struct {
  // Integer registers.
  IREE_ALIGNAS(16) int32_t i32[IREE_I32_REGISTER_COUNT];
  // Reference counted registers.
  iree_vm_ref_t ref[IREE_REF_REGISTER_COUNT];
  // Total number of valid ref registers used by the function.
  // TODO(benvanik): make the above dynamic and include i32 count.
  int8_t ref_register_count;
} iree_vm_registers_t;

// A variable-length list of registers.
//
// This structure is an overlay for the bytecode that is serialized in a
// matching format, though it can be stack allocated as needed.
typedef struct {
  uint8_t size;
  uint8_t registers[];
} iree_vm_register_list_t;
static_assert(alignof(iree_vm_register_list_t) == 1,
              "Expecting byte alignment (to avoid padding)");
static_assert(offsetof(iree_vm_register_list_t, registers) == 1,
              "Expect no padding in the struct");

// A single stack frame within the VM.
typedef struct iree_vm_stack_frame {
  // Function that the stack frame is within.
  iree_vm_function_t function;
  // Cached module state pointer for the module containing |function|.
  iree_vm_module_state_t* module_state;
  // Offset within the function.
  iree_vm_source_offset_t offset;
  // Registers used within the frame.
  // TODO(benvanik): pointer to an arena? avoids fixed overheads.
  iree_vm_registers_t registers;

  // Pointer to a register list where callers can source their return registers.
  // If omitted then the return values are assumed to be left-aligned in the
  // register banks.
  const iree_vm_register_list_t* return_registers;
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
typedef struct iree_vm_stack {
  // TODO(benvanik): add globally useful things (instance/device manager?)
  // Depth of the stack, in frames. 0 indicates an empty stack.
  int32_t depth;
  // [0-depth) valid stack frames.
  iree_vm_stack_frame_t frames[IREE_MAX_STACK_DEPTH];

  // Resolves a module to a module state within a context.
  // This will be called on function entry whenever module transitions occur.
  iree_vm_state_resolver_t state_resolver;
} iree_vm_stack_t;

// Constructs a stack in-place in |out_stack|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_init(
    iree_vm_state_resolver_t state_resolver, iree_vm_stack_t* out_stack);

// Destructs |stack|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_deinit(iree_vm_stack_t* stack);

// Returns the current stack frame or nullptr if the stack is empty.
IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_current_frame(iree_vm_stack_t* stack);

// Returns the parent stack frame or nullptr if the stack is empty.
IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_parent_frame(iree_vm_stack_t* stack);

// Enters into the given |function| and returns the callee stack frame.
// Callers must populate the argument registers as defined by the VM API.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_function_enter(
    iree_vm_stack_t* stack, iree_vm_function_t function,
    iree_vm_stack_frame_t** out_callee_frame);

// Leaves the current stack frame.
// Callers must have retrieved the result registers as defined by the VM API.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_function_leave(iree_vm_stack_t* stack);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_STACK_H_
