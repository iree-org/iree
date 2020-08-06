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

#ifndef IREE_VM_MODULE_H_
#define IREE_VM_MODULE_H_

#include <stdint.h>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_module iree_vm_module_t;
typedef struct iree_vm_stack iree_vm_stack_t;
typedef struct iree_vm_stack_frame iree_vm_stack_frame_t;

// An opaque offset into a source map that a source resolver can calculate.
// Do not assume that iree_vm_source_offset_t+1 means the next byte offset as
// backends are free to treat these as everything from pointers to machine code
// to hash codes.
typedef int64_t iree_vm_source_offset_t;

// A variable-length list of registers.
//
// This structure is an overlay for the bytecode that is serialized in a
// matching format, though it can be stack allocated as needed.
typedef struct {
  uint16_t size;
  uint16_t registers[];
} iree_vm_register_list_t;
static_assert(iree_alignof(iree_vm_register_list_t) == 2,
              "Expecting byte alignment (to avoid padding)");
static_assert(offsetof(iree_vm_register_list_t, registers) == 2,
              "Expect no padding in the struct");

// Describes the type of a function reference.
enum iree_vm_function_linkage_e {
  // Function is internal to the module and may not be reflectable.
  IREE_VM_FUNCTION_LINKAGE_INTERNAL = 0,
  // Function is an import from another module.
  IREE_VM_FUNCTION_LINKAGE_IMPORT = 1,
  // Function is an export from the module.
  IREE_VM_FUNCTION_LINKAGE_EXPORT = 2,
  // TODO(#1979): add linkage types for well-known functions like __init.
};
typedef uint16_t iree_vm_function_linkage_t;

// A function reference that can be used with the iree_vm_function_* methods.
// These should be treated as opaque and the accessor functions should be used
// instead.
//
// The register counts specify required internal storage used for VM for stack
// frame management and debugging. They must at least be able to contain all
// entry arguments for the function. The counts may be omitted if the function
// will not be referenced by a VM stack frame.
typedef struct {
  // Module the function is contained within.
  iree_vm_module_t* module;
  // Linkage of the function. Note that IREE_VM_FUNCTION_LINKAGE_INTERNAL
  // functions may be missing reflection information.
  iree_vm_function_linkage_t linkage;
  // Ordinal within the module in the linkage scope.
  uint16_t ordinal;

  // TODO(#1979): move register info to iree_vm_function_signature_t.
  // Total number of valid i32 registers used by the function.
  uint16_t i32_register_count;
  // Total number of valid ref registers used by the function.
  uint16_t ref_register_count;
} iree_vm_function_t;
static_assert(sizeof(iree_vm_function_t) == 8 + sizeof(void*),
              "Must remain small as stored on the stack");

// Describes the expected calling convention and arguments/results of a
// function.
typedef struct {
  // TODO(#1979): rework to add useful information here (types, etc).
  // Total number of arguments to the function.
  iree_host_size_t argument_count;
  // Total number of results from the function.
  iree_host_size_t result_count;
} iree_vm_function_signature_t;

// Describes the imports, exports, and capabilities of a module.
typedef struct {
  // Total number of imported functions.
  iree_host_size_t import_function_count;
  // Total number of exported functions.
  iree_host_size_t export_function_count;
  // Total number of internal functions, if debugging info is present and they
  // can be queried.
  iree_host_size_t internal_function_count;
} iree_vm_module_signature_t;

// Internal storage for the module state.
// Thread-compatible; it's expected that only one thread at a time is executing
// VM functions and accessing this state.
typedef struct iree_vm_module_state iree_vm_module_state_t;

// Function call arguments.
typedef struct {
  // Function to call.
  iree_vm_function_t function;
  // Optional list of source argument registers within the caller frame.
  // The arguments will be copied or moved from the caller to the callee upon
  // function entry.
  const iree_vm_register_list_t* argument_registers;
  // Optional variadic argument segment size list that can be provided to call
  // functions with variadic arguments. Each element represents one value per
  // logical operand group in the call with non-variadic arguments having a
  // value of 1 and variadic arguments having a value in the range of 0 to N.
  const iree_vm_register_list_t* variadic_segment_size_list;
  // Optional list of target result registers within the caller frame.
  // When the target function returns the register will be copied from the
  // callee frame to the caller frame.
  const iree_vm_register_list_t* result_registers;
} iree_vm_function_call_t;

// Results of an iree_vm_module_execute request.
typedef struct {
  // TODO(benvanik): yield information.
  // Yield modes:
  // - yield (yield instruction)
  // - await (with 1+ wait handles)
  // - break
  int reserved;
} iree_vm_execution_result_t;

// Defines an interface that can be used to reflect and execute functions on a
// module.
//
// Module implementations must be thread-safe as lookups and executions may
// occur in any order from any thread.
// TODO(benvanik): version this interface.
typedef struct iree_vm_module {
  void* self;
  iree_atomic_intptr_t ref_count;

  // Destroys |self| when all references to the module have been released.
  void(IREE_API_PTR* destroy)(void* self);

  // Returns the name of the module (used during resolution).
  iree_string_view_t(IREE_API_PTR* name)(void* self);

  // Returns the reflected signature of the module.
  iree_vm_module_signature_t(IREE_API_PTR* signature)(void* self);

  // Gets one or more pieces of function information:
  // - |out_function| set to the function reference.
  // - |out_name| set to the function name.
  // - |out_signature| set to the function signature.
  iree_status_t(IREE_API_PTR* get_function)(
      void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
      iree_vm_function_t* out_function, iree_string_view_t* out_name,
      iree_vm_function_signature_t* out_signature);

  // Looks up a function with the given name and linkage in the module.
  // This may perform a linear scan and results should be cached.
  iree_status_t(IREE_API_PTR* lookup_function)(
      void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
      iree_vm_function_t* out_function);

  // Allocates module state data.
  iree_status_t(IREE_API_PTR* alloc_state)(
      void* self, iree_allocator_t allocator,
      iree_vm_module_state_t** out_module_state);

  // Frees module state data.
  void(IREE_API_PTR* free_state)(void* self,
                                 iree_vm_module_state_t* module_state);

  // Resolves the import with the given ordinal to |function|.
  // The function is guaranteed to remain valid for the lifetime of the module
  // state.
  iree_status_t(IREE_API_PTR* resolve_import)(
      void* self, iree_vm_module_state_t* module_state,
      iree_host_size_t ordinal, iree_vm_function_t function);

  // Begins a function call with the given |call| arguments.
  // Execution may yield in the case of asynchronous code and require one or
  // more calls to the resume method to complete.
  iree_status_t(IREE_API_PTR* begin_call)(
      void* self, iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
      iree_vm_execution_result_t* out_result);

  // Resumes execution of a previously-yielded call.
  iree_status_t(IREE_API_PTR* resume_call)(
      void* self, iree_vm_stack_t* stack,
      iree_vm_execution_result_t* out_result);

  // TODO(benvanik): move this/refactor.
  // Gets a reflection attribute for a function by index.
  // The returned key and value strings are guaranteed valid for the life
  // of the module. Note that not all modules and functions have reflection
  // attributes.
  // Returns IREE_STATUS_NOT_FOUND if index >= the number of attributes for
  // the function.
  // See: docs/design_docs/function_abi.md
  iree_status_t(IREE_API_PTR* get_function_reflection_attr)(
      void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
      iree_host_size_t index, iree_string_view_t* key,
      iree_string_view_t* value);
} iree_vm_module_t;

// Initializes the interface of a module handle.
// This should be called by module implementations after they allocate
// themselves to properly initialize the module interface prior to populating
// interface function pointers. This ensures that version adaptation can be
// performed by the library as needed.
// TODO(benvanik): version/module size.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_initialize(iree_vm_module_t* module, void* self);

// Retains the given |module| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_module_retain(iree_vm_module_t* module);

// Releases the given |module| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_module_release(iree_vm_module_t* module);

// Returns the name of the module (used during resolution).
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_module_name(const iree_vm_module_t* module);

// Returns the signature of the module describing the contents.
IREE_API_EXPORT iree_vm_module_signature_t IREE_API_CALL
iree_vm_module_signature(const iree_vm_module_t* module);

// Looks up a function with the given name and linkage in the |module|.
// This may perform a linear scan and results should be cached.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_lookup_function_by_name(const iree_vm_module_t* module,
                                       iree_vm_function_linkage_t linkage,
                                       iree_string_view_t name,
                                       iree_vm_function_t* out_function);

// Looks up a function with the given ordinal and linkage in the |module|.
// If |linkage_name| is not null, then it will be populated with the name
// of the linkage record (i.e. the actual exported name vs the internal
// name which would be returned in a subsequent call to iree_vm_function_name).
// TODO(laurenzo): Remove out_linkage_name in favore of a LINKAGE_PUBLIC (with
// the name that you'd get from a function_name call on that being the public
// name).
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_lookup_function_by_ordinal(const iree_vm_module_t* module,
                                          iree_vm_function_linkage_t linkage,
                                          iree_host_size_t ordinal,
                                          iree_vm_function_t* out_function,
                                          iree_string_view_t* out_linkage_name);

// Returns the name of the given function or empty string if not available.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_function_name(const iree_vm_function_t* function);

// Returns the signature of the function if reflection metadata is available.
IREE_API_EXPORT iree_vm_function_signature_t IREE_API_CALL
iree_vm_function_signature(const iree_vm_function_t* function);

// Returns a value for the given reflection attribute |key|, if found.
// Returns the empty string if the reflection data in general or the specific
// key is not found.
//
// See: docs/design_docs/function_abi.md for documentation on the ABI.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_function_reflection_attr(const iree_vm_function_t* function,
                                 iree_string_view_t key);

// TODO(#1979): remove this and use iree_vm_function_reflection_attr.
// Gets a reflection attribute for a function by index.
// The returned key and value strings are guaranteed valid for the life
// of the module. Note that not all modules and functions have reflection
// attributes.
// Returns IREE_STATUS_NOT_FOUND if index >= the number of attributes for
// the function.
// See: docs/design_docs/function_abi.md
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_get_function_reflection_attr(iree_vm_function_t function,
                                     iree_host_size_t index,
                                     iree_string_view_t* key,
                                     iree_string_view_t* value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_MODULE_H_
