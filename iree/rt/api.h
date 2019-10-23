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

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_RT_API_H_
#define IREE_RT_API_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

typedef struct iree_rt_instance iree_rt_instance_t;
typedef struct iree_rt_context iree_rt_context_t;
typedef struct iree_rt_policy iree_rt_policy_t;
typedef struct iree_rt_module iree_rt_module_t;
typedef struct iree_rt_invocation iree_rt_invocation_t;

// Describes the type of a function reference.
typedef enum {
  // Function is internal to the module and may not be reflectable.
  IREE_RT_FUNCTION_LINKAGE_INTERNAL = 0,
  // Function is an import from another module.
  IREE_RT_FUNCTION_LINKAGE_IMPORT = 1,
  // Function is an export from the module.
  IREE_RT_FUNCTION_LINKAGE_EXPORT = 2,
} iree_rt_function_linkage_t;

// A function reference that can be used with the iree_rt_function_* methods.
// These should be treated as opaque and the accessor functions should be used
// instead.
typedef struct {
  // Module the function is contained within.
  iree_rt_module_t* module;
  // Linkage of the function. Note that IREE_RT_FUNCTION_LINKAGE_INTERNAL
  // functions may be missing reflection information.
  iree_rt_function_linkage_t linkage;
  // Ordinal within the module in the linkage scope.
  int32_t ordinal;
} iree_rt_function_t;

// Describes the expected calling convention and arguments/results of a
// function.
typedef struct {
  // Total number of arguments to the function.
  int32_t argument_count;
  // Total number of results from the function.
  int32_t result_count;
} iree_rt_function_signature_t;

// Describes the imports, exports, and capabilities of a module.
typedef struct {
  // Total number of imported functions.
  int32_t import_function_count;
  // Total number of exported functions.
  int32_t export_function_count;
  // Total number of internal functions, if debugging info is present and they
  // can be queried.
  int32_t internal_function_count;
  // Total number of state block resource slots consumed.
  int32_t state_slot_count;
} iree_rt_module_signature_t;

// Dependency information used to order invocations.
typedef struct {
  // Prior invocations that must complete before the new invocation begins.
  iree_rt_invocation_t** invocations;
  iree_host_size_t invocation_count;

  // TODO(benvanik): wait semaphores/importing.
} iree_rt_invocation_dependencies_t;

// Defines an external module that can be used to reflect and execute functions.
// Modules must be thread-safe as lookups and executions may occur in any order
// from any thread.
//
// Modules will have their resolve_imports function called upon registration
// with a context and may use the provided resolver to find imported functions.
typedef struct {
  // User-defined pointer passed to all functions.
  void* self;
  // Destroys |self| when all references to the module have been released.
  iree_status_t(IREE_API_PTR* destroy)(void* self);
  // Returns the name of the module (used during resolution).
  iree_string_view_t(IREE_API_PTR* name)(void* self);
  // Sets |out_module_signature| to the reflected signature of the module.
  iree_rt_module_signature_t(IREE_API_PTR* signature)(void* self);
  // Sets |out_function| to a resolved function by ordinal, if found.
  iree_status_t(IREE_API_PTR* lookup_function_by_ordinal)(
      void* self, iree_rt_function_linkage_t linkage, int32_t ordinal,
      iree_rt_function_t* out_function);
  // Sets |out_function| to a resolved function by name, if found.
  iree_status_t(IREE_API_PTR* lookup_function_by_name)(
      void* self, iree_rt_function_linkage_t linkage, iree_string_view_t name,
      iree_rt_function_t* out_function);
  // Sets |out_name| to the name of the function with the given ordinal, if
  // found.
  iree_status_t(IREE_API_PTR* get_function_name)(
      void* self, iree_rt_function_linkage_t linkage, int32_t ordinal,
      iree_string_view_t* out_name);
  // Sets |out_signature| to the reflected signature of the given
  // function, if found.
  iree_status_t(IREE_API_PTR* get_function_signature)(
      void* self, iree_rt_function_linkage_t linkage, int32_t ordinal,
      iree_rt_function_signature_t* out_signature);
} iree_rt_external_module_t;

//===----------------------------------------------------------------------===//
// iree::rt::Instance
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a new instance. This should be shared with all contexts in an
// application to ensure that resources are tracked properly and threads are
// managed correctly.
// |out_instance| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_instance_create(
    iree_allocator_t allocator, iree_rt_instance_t** out_instance);

// Retains the given |instance| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_instance_retain(iree_rt_instance_t* instance);

// Releases the given |instance| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_instance_release(iree_rt_instance_t* instance);

// TEMPORARY: until policies and placement are performed this can be used to
// explicitly create and register drivers by name.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_instance_register_driver_ex(
    iree_rt_instance_t* instance, iree_string_view_t driver_name);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::rt::Module
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a module with an external backing implementation.
// The provided |external_module| definition will be used to query the module
// state as needed. No caching occurs within the implementation to allow calls
// to return different values per-invocation.
//
// |out_module| must be released by the caller.
// iree_rt_external_module_t::destroy is called when the last reference to the
// iree_rt_module_t is released.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_module_create_external(
    iree_rt_external_module_t impl, iree_allocator_t allocator,
    iree_rt_module_t** out_module);

// Retains the given |module| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_retain(iree_rt_module_t* module);

// Releases the given |module| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_release(iree_rt_module_t* module);

// Returns the name of the module.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_rt_module_name(const iree_rt_module_t* module);

// Sets |out_function| to a function with |ordinal| in the given linkage or
// returns IREE_STATUS_NOT_FOUND. The function reference is valid for the
// lifetime of |module|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_lookup_function_by_ordinal(iree_rt_module_t* module,
                                          iree_rt_function_linkage_t linkage,
                                          int32_t ordinal,
                                          iree_rt_function_t* out_function);

// Sets |out_function| to a function with |name| in the given linkage or returns
// IREE_STATUS_NOT_FOUND. The function reference is valid for the lifetime of
// |module|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_lookup_function_by_name(iree_rt_module_t* module,
                                       iree_rt_function_linkage_t linkage,
                                       iree_string_view_t name,
                                       iree_rt_function_t* out_function);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::rt::Function
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Returns the name of the function as exported from the module.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_rt_function_name(const iree_rt_function_t* function);

// Sets |out_function_signature| to the reflected signature of the function.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_function_signature(const iree_rt_function_t* function,
                           iree_rt_function_signature_t* out_signature);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::rt::Policy
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// TODO(benvanik): define policies. For now they are no-ops.
IREE_API_EXPORT iree_status_t iree_rt_policy_create(
    iree_allocator_t allocator, iree_rt_policy_t** out_policy);

// Retains the given |policy| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_policy_retain(iree_rt_policy_t* policy);

// Releases the given |policy| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_policy_release(iree_rt_policy_t* policy);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::rt::Context
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a new context that uses the given |instance| for device management.
// |out_context| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_context_create(
    iree_rt_instance_t* instance, iree_rt_policy_t* policy,
    iree_allocator_t allocator, iree_rt_context_t** out_context);

// Retains the given |context| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_retain(iree_rt_context_t* context);

// Releases the given |context| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_release(iree_rt_context_t* context);

// Returns a process-unique ID for the |context|.
IREE_API_EXPORT int32_t IREE_API_CALL
iree_rt_context_id(const iree_rt_context_t* context);

// Registers a list of modules with the context and resolves imports.
// The modules will be retained by the context until destruction.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_context_register_modules(
    iree_rt_context_t* context, iree_rt_module_t** modules,
    iree_host_size_t module_count);

// Returns a reference to the module registered with the given name or nullptr
// if not found. The caller must retain the returned module if they want to
// continue using it.
IREE_API_EXPORT iree_rt_module_t* IREE_API_CALL
iree_rt_context_lookup_module_by_name(const iree_rt_context_t* context,
                                      iree_string_view_t module_name);

// Sets |out_function| to to an exported function with the fully-qualified name
// of |full_name| or returns IREE_STATUS_NOT_FOUND. The function reference is
// valid for the lifetime of |context|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_context_resolve_function(
    const iree_rt_context_t* context, iree_string_view_t full_name,
    iree_rt_function_t* out_function);

// Allocates a host-local buffer that is optimal for use on the host but is
// usable by the given |device_placements| (at a possible performance
// penalty). The buffer can be used for staging uploads to device-local
// buffers and is useful for times when the buffer will be used more on the
// host than the device. If a buffer never needs to be used with a device
// prefer instead HeapBuffer::Allocate.
//
// Fails if it is not possible to allocate and satisfy all placements for the
// requested |buffer_usage|.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_allocate_device_visible_buffer(
    iree_rt_context_t* context, iree_hal_buffer_usage_t buffer_usage,
    iree_host_size_t allocation_size, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer);

// Allocates a device-local buffer that is optimal for use with the given
// |device_placements|. The buffer will not be host-visible and can only be
// used from compatible device queues.
//
// Fails if it is not possible to allocate and satisfy all placements for the
// requested |buffer_usage|.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_allocate_device_local_buffer(
    iree_rt_context_t* context, iree_hal_buffer_usage_t buffer_usage,
    iree_host_size_t allocation_size, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::rt::Invocation
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a new invocation tracking object for invoking the given |function|
// from |context|. |arguments| will be retained until the invocation is made.
// If |dependencies| are provided then the invocation will wait until they are
// resolved before executing. If a |policy| is provided it will override the
// context-level policy.
//
// Optionally |results| may be provided with preallocated buffers that will
// receive the outputs of the invocation. Invocation will fail if they do not
// match expected sizes.
//
// Note that it's possible for the invocation to complete prior to the return of
// this function. Any errors that occur will be set on the invocation and
// callers should query its state prior to assuming it is in-flight.
//
// |out_invocation| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_invocation_create(
    iree_rt_context_t* context, iree_rt_function_t* function,
    iree_rt_policy_t* policy,
    const iree_rt_invocation_dependencies_t* dependencies,
    iree_hal_buffer_view_t** arguments, iree_host_size_t argument_count,
    iree_hal_buffer_view_t** results, iree_host_size_t result_count,
    iree_allocator_t allocator, iree_rt_invocation_t** out_invocation);

// Retains the given |invocation| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_retain(iree_rt_invocation_t* invocation);

// Releases the given |invocation| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_release(iree_rt_invocation_t* invocation);

// Queries the completion status of the invocation.
// Returns one of the following:
//   IREE_STATUS_OK: the invocation completed successfully.
//   IREE_STATUS_UNAVAILABLE: the invocation has not yet completed.
//   IREE_STATUS_CANCELLED: the invocation was cancelled internally.
//   IREE_STATUS_ABORTED: the invocation was aborted.
//   IREE_STATUS_*: an error occurred during invocation.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_query_status(iree_rt_invocation_t* invocation);

// Populates |out_results| to the values of the results.
// |result_capacity| defines the number of elements available in |out_results|
// and |out_result_count| will be set with the actual number of results
// available. If |result_capacity| is too small IREE_STATUS_OUT_OF_RANGE will be
// returned wtih the required capacity in |out_result_count|. To only query the
// required capacity |out_results| may be passed as nullptr.
//
// Ownership of returned results will be transferred to the caller and they must
// be released if no longer needed.
//
// Returns errors as with iree_rt_invocation_query_status, for example in the
// case of not-yet-completed or aborted invocations.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_invocation_consume_results(
    iree_rt_invocation_t* invocation, iree_host_size_t result_capacity,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_results,
    iree_host_size_t* out_result_count);

// Blocks the caller until the invocation completes (successfully or otherwise).
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |deadline| elapses before the
// invocation completes and otherwise returns iree_rt_invocation_query_status.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_invocation_await(
    iree_rt_invocation_t* invocation, iree_time_t deadline);

// Attempts to abort the invocation if it is in-flight.
// A no-op if the invocation has already completed.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_abort(iree_rt_invocation_t* invocation);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_RT_API_H_
