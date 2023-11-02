// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_H_
#define IREE_VM_MODULE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_module_t iree_vm_module_t;
typedef struct iree_vm_stack_t iree_vm_stack_t;
typedef struct iree_vm_stack_frame_t iree_vm_stack_frame_t;

//===----------------------------------------------------------------------===//
// Module / function reflection
//===----------------------------------------------------------------------===//

// Describes the type of a function reference.
typedef enum iree_vm_function_linkage_e {
  // Function is internal to the module and may not be reflectable.
  IREE_VM_FUNCTION_LINKAGE_INTERNAL = 0,
  // Function is an import from another module.
  IREE_VM_FUNCTION_LINKAGE_IMPORT = 1,
  // Function is an export from the module.
  IREE_VM_FUNCTION_LINKAGE_EXPORT = 2,
  // Function is an import from another module that may be unavailable.
  IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL = 3,
} iree_vm_function_linkage_t;

// A function reference that can be used with the iree_vm_function_* methods.
// These should be treated as opaque and the accessor functions should be used
// instead.
//
// The register counts specify required internal storage used for VM for stack
// frame management and debugging. They must at least be able to contain all
// entry arguments for the function. The counts may be omitted if the function
// will not be referenced by a VM stack frame.
typedef struct iree_vm_function_t {
  // Module the function is contained within.
  iree_vm_module_t* module;
  // Linkage of the function. Note that IREE_VM_FUNCTION_LINKAGE_INTERNAL
  // functions may be missing reflection information.
  uint16_t linkage;
  // Ordinal within the module in the linkage scope.
  uint16_t ordinal;
} iree_vm_function_t;
static_assert(sizeof(iree_vm_function_t) <= 3 * sizeof(void*),
              "Must remain small as stored on the stack");

// Returns true if the |function| is null (didn't exist, etc).
static inline bool iree_vm_function_is_null(iree_vm_function_t function) {
  return function.module == NULL;
}

// Describes the expected calling convention and arguments/results of a
// function.
typedef struct iree_vm_function_signature_t {
  // The VM calling convention declaration used to marshal arguments and
  // results into and out of the function.
  // Optional for imports and internal functions but required for exports.
  //
  // Format:
  // - '0': version 0 prefix
  // - Zero or more arguments:
  //   - 'i': int32_t integer (i32)
  //   - 'I': int64_t integer (i64)
  //   - 'r': ref-counted type pointer (!vm.ref<?>)
  //   - 'C' ... 'D': variadic list of flattened tuples of a specified type
  // - EOL or '_'
  // - Zero or more results:
  //   - 'i' or 'I'
  //   - 'r'
  //
  // Examples:
  //   `0` or `0_`: () -> ()
  //   `0i` or `0i_`: (i32) -> ()
  //   `0iiCiiD_i`: (i32, i32, tuple<i32, i32>...) -> i32
  //   `0irCirD_r`: (i32, !vm.ref<?>, tuple<i32, !vm.ref<?>>) -> !vm.ref<?>
  //
  // Users of this field must verify the version prefix in the first byte before
  // using the declaration.
  iree_string_view_t calling_convention;
} iree_vm_function_signature_t;

// Describes the imports, exports, and capabilities of a module.
typedef struct iree_vm_module_signature_t {
  // Module version.
  uint32_t version;
  // Total number of module-level attributes.
  iree_host_size_t attr_count;
  // Total number of imported functions.
  iree_host_size_t import_function_count;
  // Total number of exported functions.
  iree_host_size_t export_function_count;
  // Total number of internal functions, if debugging info is present and they
  // can be queried.
  iree_host_size_t internal_function_count;
} iree_vm_module_signature_t;

// Defines the behavior of dependency resolution.
enum iree_vm_module_dependency_flag_bits_t {
  IREE_VM_MODULE_DEPENDENCY_FLAG_NONE = 0u,
  // Module is required and must have the minimum version specified.
  // Individual imports may still be optional.
  IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED = 1u << 0,
  // Module is optional and if not present the module will still load. All
  // methods imported from the module must also be marked optional.
  IREE_VM_MODULE_DEPENDENCY_FLAG_OPTIONAL = 1u << 1,
};
typedef uint32_t iree_vm_module_dependency_flags_t;

// Describes a module's dependency on another module.
typedef struct iree_vm_module_dependency_t {
  // Name of the module (`hal`, etc).
  iree_string_view_t name;
  // Minimum required version in order to satisfy the dependency.
  uint32_t minimum_version;
  // Flags controlling the dependency resolution behavior.
  iree_vm_module_dependency_flags_t flags;
} iree_vm_module_dependency_t;

// Callback issued as part of module dependency enumeration.
// The dependency and references such as the module name are only valid for the
// duration of the callback and callers must copy them out in order to extend
// their lifetime.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_dependency_callback_t)(
    void* user_data_ptr, const iree_vm_module_dependency_t* dependency);

// Internal storage for the module state.
// Thread-compatible; it's expected that only one thread at a time is executing
// VM functions and accessing this state.
typedef struct iree_vm_module_state_t iree_vm_module_state_t;

//===----------------------------------------------------------------------===//
// Function ABI management
//===----------------------------------------------------------------------===//

// A variable-length list of registers.
//
// This structure is an overlay for the bytecode that is serialized in a
// matching format, though it can be stack allocated as needed.
//
// TODO(benvanik): this should be made private to the bytecode module, but is
// used for toll-free variadic argument lists here. We could just define an
// identical structure (and static_assert) to at least rename it to something
// sensible (iree_vm_segment_size_list_t).
typedef struct iree_vm_register_list_t {
  uint16_t size;
  uint16_t registers[];
} iree_vm_register_list_t;
static_assert(iree_alignof(iree_vm_register_list_t) == 2,
              "expecting byte alignment (to avoid padding)");
static_assert(offsetof(iree_vm_register_list_t, registers) == 2,
              "expect no padding in the struct");

// Function call data.
//
// Arguments and results are encoded following a standard format shared across
// all module types. This allows implementations that have different storage
// types (such as physical machine registers vs. virtual registers) to use the
// same cross-module calling convention.
//
// Callees can assume that callers have properly allocated and setup the
// argument and result buffers and need not verify them. This works only because
// the calling convention format is directly queried from the callee module.
//
// Encoding:
// - each int is encoded as a 4-byte aligned value
// - each ref is encoded as a 4-byte aligned iree_vm_ref_t value
// - variadic tuples are encoded as a 4-byte count prefix and the tuple values
//
// For example, (i32, tuple<!vm.ref<?>, i32>..., i32) is encoded as:
//    4b: i32
//    4b: tuple count
//    repeated:
//      8b-16b: iree_vm_ref_t
//      4b: i32
//    4b: i32
//
// Example sequence:
//  1. ModuleA wants to call SomeFunction from ModuleB
//  2. ModuleA imports SomeFunction from ModuleB and gets its
//     iree_vm_function_signature_t during import resolution
//  3. ModuleA checks that it understands/supports that calling convention
//     with error handling if needed (e.g. if ModuleB is newer and uses a newer
//     version that ModuleA wasn't compiled knowing about, or ModuleB is ancient
//     and uses a deprecated version that ModuleA has already dropped)
//  4. ModuleA prepares argument and result buffers according to the calling
//     convention defined by ModuleB and calls SomeFunction
//  5. ModuleB handles the call, trusting that the input and output buffers are
//     as expected
//
// NOTE: we could switch to using libffi, but I didn't want to require that for
// all uses and didn't want to enable the issues that can arise when crossing
// device boundaries. With what we have here we can rather easily serialize the
// argument/result buffers and map them between independent address spaces.
// Instead, implementing a native_module-alike of libffi_module would be a
// better layering for callee modules.
typedef struct iree_vm_function_call_t {
  // Function to call.
  iree_vm_function_t function;

  // Argument buffer in the format described above.
  // This is only read on beginning the function and need not live beyond that.
  //
  // Refs contained are retained by the caller and callees must retain them if
  // they need them to live beyond the call.
  iree_byte_span_t arguments;

  // Storage for the result buffer; assumed undefined and then populated with
  // data in a format described above. This is required for both the beginning
  // of function invocation as well as each resume (as any may actually return
  // control flow).
  //
  // Refs contained will be retained in the results buffer and callers must
  // either move or release them upon return from the call.
  iree_byte_span_t results;
} iree_vm_function_call_t;

#define IREE_VM_CCONV_TYPE_VOID 'v'
#define IREE_VM_CCONV_TYPE_I32 'i'
#define IREE_VM_CCONV_TYPE_I64 'I'
#define IREE_VM_CCONV_TYPE_F32 'f'
#define IREE_VM_CCONV_TYPE_F64 'F'
#define IREE_VM_CCONV_TYPE_REF 'r'
#define IREE_VM_CCONV_TYPE_SPAN_START 'C'
#define IREE_VM_CCONV_TYPE_SPAN_END 'D'

// Returns the arguments and results fragments from the function signature.
// Either may be empty if they have no values.
//
// Example:
//  ``          -> arguments = ``, results = ``
//  `0`         -> arguments = ``, results = ``
//  `0v`        -> arguments = ``, results = ``
//  `0ri`       -> arguments = `ri`, results = ``
//  `0_ir`      -> arguments = ``, results = `ir`
//  `0v_ir`     -> arguments = ``, results = `ir`
//  `0iCiD_rr`  -> arguments = `iCiD`, results = `rr`
IREE_API_EXPORT iree_status_t iree_vm_function_call_get_cconv_fragments(
    const iree_vm_function_signature_t* signature,
    iree_string_view_t* out_arguments, iree_string_view_t* out_results);

// Returns true if the given cconv contains one or more variadic types.
IREE_API_EXPORT bool iree_vm_function_call_is_variadic_cconv(
    iree_string_view_t cconv);

// Counts the total number of arguments and results of a function.
IREE_API_EXPORT iree_status_t iree_vm_function_call_count_arguments_and_results(
    const iree_vm_function_signature_t* signature,
    iree_host_size_t* out_argument_count, iree_host_size_t* out_result_count);

// Returns the required size, in bytes, to store the data in the given cconv
// fragment (like `iICriDr`).
//
// The provided |segment_size_list| is used for variadic arguments/results. Each
// entry represents one of the top level arguments with spans being flattened.
IREE_API_EXPORT iree_status_t iree_vm_function_call_compute_cconv_fragment_size(
    iree_string_view_t cconv_fragment,
    const iree_vm_register_list_t* segment_size_list,
    iree_host_size_t* out_required_size);

//===----------------------------------------------------------------------===//
// Source locations
//===----------------------------------------------------------------------===//

// An opaque offset into a source map that a source resolver can calculate.
// Do not assume that iree_vm_source_offset_t+1 means the next byte offset as
// backends are free to treat these as everything from pointers to machine code
// to hash codes.
typedef int64_t iree_vm_source_offset_t;

// Controls how source locations are formatted into strings.
enum iree_vm_source_location_format_flag_bits_e {
  IREE_VM_SOURCE_LOCATION_FORMAT_FLAG_NONE = 0u,
  // Only formats a single line (excluding \n) for the source location, even
  // if the full location information (such as a backtrace) is available.
  IREE_VM_SOURCE_LOCATION_FORMAT_FLAG_SINGLE_LINE = 1u << 0,
};
typedef uint32_t iree_vm_source_location_format_flags_t;

// Source location interface.
typedef struct iree_vm_source_location_t {
  IREE_API_UNSTABLE

  // Implementation-specified fields. Do not use directly.
  void* self;
  uint64_t data[2];

  iree_status_t(IREE_API_PTR* format)(
      void* self, uint64_t data[2],
      iree_vm_source_location_format_flags_t flags,
      iree_string_builder_t* builder);
} iree_vm_source_location_t;

// Formats the |source_location| to its canonical string form.
IREE_API_EXPORT iree_status_t
iree_vm_source_location_format(iree_vm_source_location_t* source_location,
                               iree_vm_source_location_format_flags_t flags,
                               iree_string_builder_t* builder);

//===----------------------------------------------------------------------===//
// iree_vm_module_t
//===----------------------------------------------------------------------===//

// Indicates an event that can be signaled in modules from the hosting program.
typedef enum iree_vm_signal_e {
  // Program is resuming from a suspended state.
  // Modules may reallocate memory for pools and caches.
  //
  // Modules are walked in registration order (A->B->C).
  IREE_VM_SIGNAL_RESUME = 0,

  // Program is entering a suspended state.
  // Modules should drop any transient memory that is possible to reallocate
  // upon resume.
  //
  // Modules are walked in reverse registration order (C->B->A).
  IREE_VM_SIGNAL_SUSPEND = 1,

  // Program has received a low memory alert.
  // Modules must aggressively drop all possible memory even if expensive to
  // rematerialize it. On some platforms this is sent as a threat that if
  // sufficient memory is not unwired/freed ASAP the process will be killed.
  //
  // Modules are walked in reverse registration order (C->B->A).
  IREE_VM_SIGNAL_LOW_MEMORY = 2,
} iree_vm_signal_t;

// Defines an interface that can be used to reflect and execute functions on a
// module.
//
// Module implementations must be thread-safe as lookups and executions may
// occur in any order from any thread.
// TODO(benvanik): version this interface.
typedef struct iree_vm_module_t {
  IREE_API_UNSTABLE

  void* self;
  iree_atomic_ref_count_t ref_count;

  // Destroys |self| when all references to the module have been released.
  void(IREE_API_PTR* destroy)(void* self);

  // Returns the name of the module (used during resolution).
  iree_string_view_t(IREE_API_PTR* name)(void* self);

  // Returns the reflected signature of the module.
  iree_vm_module_signature_t(IREE_API_PTR* signature)(void* self);

  // Enumerates all module dependencies.
  iree_status_t(IREE_API_PTR* enumerate_dependencies)(
      void* self, iree_vm_module_dependency_callback_t callback,
      void* user_data);

  // Gets a reflection attribute for the module by attribute index.
  // The returned key and value strings are guaranteed valid for the life
  // of the module. Note that not all modules have reflection attributes.
  // Returns IREE_STATUS_OUT_OF_RANGE if index >= the number of attributes.
  iree_status_t(IREE_API_PTR* get_module_attr)(void* self,
                                               iree_host_size_t index,
                                               iree_string_pair_t* out_attr);

  // Looks up a function with the given name and linkage in the module.
  // This may perform a linear scan and results should be cached.
  // An optional |expected_signature| can be specified in cases where the
  // module may be able to provide additional validation or versioning based on
  // it. Implementations should not assume all lookups will include a signature.
  iree_status_t(IREE_API_PTR* lookup_function)(
      void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
      const iree_vm_function_signature_t* expected_signature,
      iree_vm_function_t* out_function);

  // Gets one or more pieces of function information:
  // - |out_function| set to the function reference.
  // - |out_name| set to the function name.
  // - |out_signature| set to the function signature.
  iree_status_t(IREE_API_PTR* get_function)(
      void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
      iree_vm_function_t* out_function, iree_string_view_t* out_name,
      iree_vm_function_signature_t* out_signature);

  // Gets a reflection attribute for a function by attribute index.
  // The returned key and value strings are guaranteed valid for the life
  // of the module. Note that not all modules and functions have reflection
  // attributes.
  // Returns IREE_STATUS_OUT_OF_RANGE if index >= the number of attributes for
  // the function.
  iree_status_t(IREE_API_PTR* get_function_attr)(
      void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
      iree_host_size_t index, iree_string_pair_t* out_attr);

  // Resolves a |function| at |pc| from the module to a |out_source_location|,
  // if debug information is available.
  iree_status_t(IREE_API_PTR* resolve_source_location)(
      void* self, iree_vm_function_t function, iree_vm_source_offset_t pc,
      iree_vm_source_location_t* out_source_location);

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
      iree_host_size_t ordinal, const iree_vm_function_t* function,
      const iree_vm_function_signature_t* signature);

  // Notifies the module of a system signal.
  iree_status_t(IREE_API_PTR* notify)(void* self,
                                      iree_vm_module_state_t* module_state,
                                      iree_vm_signal_t signal);

  // Begins a function call with the given |call| arguments.
  //
  // Returns OK if execution completes immediately. If the call completes
  // immediately the results will be written to |call|->results.
  //
  // Returns IREE_STATUS_DEFERRED if execution yielded and the call needs to be
  // resumed. Depending on the program it may be unsafe to begin any other calls
  // without first completing prior ones.
  iree_status_t(IREE_API_PTR* begin_call)(void* self, iree_vm_stack_t* stack,
                                          iree_vm_function_call_t call);

  // Resumes execution of a previously-yielded call.
  //
  // Returns OK if execution completes immediately. If the call completes
  // immediately the results will be written to |call|->results.
  //
  // Returns IREE_STATUS_DEFERRED if execution yielded and the call needs to be
  // resumed. Depending on the program it may be unsafe to begin any other calls
  // without first completing prior ones.
  iree_status_t(IREE_API_PTR* resume_call)(void* self, iree_vm_stack_t* stack,
                                           iree_byte_span_t call_results);
} iree_vm_module_t;

// Initializes the interface of a module handle.
// This should be called by module implementations after they allocate
// themselves to properly initialize the module interface prior to populating
// interface function pointers. This ensures that version adaptation can be
// performed by the library as needed.
// TODO(benvanik): version/module size.
IREE_API_EXPORT iree_status_t
iree_vm_module_initialize(iree_vm_module_t* module, void* self);

// Retains the given |module| for the caller.
IREE_API_EXPORT void iree_vm_module_retain(iree_vm_module_t* module);

// Releases the given |module| from the caller.
IREE_API_EXPORT void iree_vm_module_release(iree_vm_module_t* module);

// Returns the name of the module (used during resolution).
IREE_API_EXPORT iree_string_view_t
iree_vm_module_name(const iree_vm_module_t* module);

// Returns the signature of the module describing the contents.
IREE_API_EXPORT iree_vm_module_signature_t
iree_vm_module_signature(const iree_vm_module_t* module);

// Returns a value for the given reflection attribute |key|, if found.
// Returns the empty string if the reflection data in general or the specific
// key is not found.
IREE_API_EXPORT iree_string_view_t iree_vm_module_lookup_attr_by_name(
    const iree_vm_module_t* module, iree_string_view_t key);

// Gets a reflection attribute for a module by index into the attribute list.
// The returned key and value strings are guaranteed valid for the life
// of the module. Note that not all modules have reflection attributes.
//
// Returns IREE_STATUS_OUT_OF_RANGE if index >= the number of attributes for
// the function.
IREE_API_EXPORT iree_status_t
iree_vm_module_get_attr(const iree_vm_module_t* module, iree_host_size_t index,
                        iree_string_pair_t* out_attr);

// Enumerates all modules that |module| depends on, calling the provided
// |callback| once per module.
// Failures from the callback will be propagated to the caller.
IREE_API_EXPORT iree_status_t iree_vm_module_enumerate_dependencies(
    iree_vm_module_t* module, iree_vm_module_dependency_callback_t callback,
    void* user_data);

// Looks up a function with the given name and linkage in the |module|.
// This may perform a linear scan and results should be cached.
IREE_API_EXPORT iree_status_t iree_vm_module_lookup_function_by_name(
    const iree_vm_module_t* module, iree_vm_function_linkage_t linkage,
    iree_string_view_t name, iree_vm_function_t* out_function);

// Looks up a function with the given ordinal and linkage in the |module|.
IREE_API_EXPORT iree_status_t iree_vm_module_lookup_function_by_ordinal(
    const iree_vm_module_t* module, iree_vm_function_linkage_t linkage,
    iree_host_size_t ordinal, iree_vm_function_t* out_function);

// Resolves a |function| at |pc| from the module to a |out_source_location|, if
// debug information is available.
IREE_API_EXPORT iree_status_t iree_vm_module_resolve_source_location(
    const iree_vm_module_t* module, iree_vm_function_t function,
    iree_vm_source_offset_t pc, iree_vm_source_location_t* out_source_location);

//===----------------------------------------------------------------------===//
// iree_vm_function_t
//===----------------------------------------------------------------------===//

// Returns the name of the given function or empty string if not available.
IREE_API_EXPORT iree_string_view_t
iree_vm_function_name(const iree_vm_function_t* function);

// Returns the signature of the function if reflection metadata is available.
IREE_API_EXPORT iree_vm_function_signature_t
iree_vm_function_signature(const iree_vm_function_t* function);

// Returns a value for the given reflection attribute |key|, if found.
// Returns the empty string if the reflection data in general or the specific
// key is not found.
IREE_API_EXPORT iree_string_view_t iree_vm_function_lookup_attr_by_name(
    const iree_vm_function_t* function, iree_string_view_t key);

// Gets a reflection attribute for a function by index into the attribute list.
// The returned key and value strings are guaranteed valid for the life
// of the module. Note that not all functions have reflection attributes.
//
// For more information on the function ABI and its reflection metadata see:
// https://iree.dev/developers/design-docs/function-abi/.
//
// Returns IREE_STATUS_OUT_OF_RANGE if index >= the number of attributes for
// the function.
//
// NOTE: always prefer to use iree_vm_function_lookup_attr_by_name; this should
// only be used when exporting attributes into a generic data structure (JSON
// or python dicts, etc).
IREE_API_EXPORT iree_status_t
iree_vm_function_get_attr(iree_vm_function_t function, iree_host_size_t index,
                          iree_string_pair_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_vm_module, iree_vm_module_t);

#endif  // IREE_VM_MODULE_H_
