// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_PARAMETER_PROVIDER_H_
#define IREE_IO_PARAMETER_PROVIDER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_parameter_provider_t
//===----------------------------------------------------------------------===//

// Indicates an event signaled from the hosting program.
typedef enum iree_io_parameter_provider_signal_e {
  // Program is resuming from a suspended state.
  // Providers may reallocate memory for pools and caches.
  IREE_IO_PARAMETER_PROVIDER_SIGNAL_RESUME = 0,
  // Program is entering a suspended state.
  // Providers should drop any transient memory that is possible to reallocate
  // upon resume.
  IREE_IO_PARAMETER_PROVIDER_SIGNAL_SUSPEND = 1,
  // Program has received a low memory alert.
  // Providers must aggressively drop all possible memory even if expensive to
  // rematerialize it. On some platforms this is sent as a threat that if
  // sufficient memory is not unwired/freed ASAP the process will be killed.
  IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY = 2,
} iree_io_parameter_provider_signal_t;

typedef struct iree_io_parameter_span_t {
  uint64_t parameter_offset;
  iree_device_size_t buffer_offset;
  iree_device_size_t length;
} iree_io_parameter_span_t;

// Interface for providers of parameter storage and caching.
// Parameters are referenced by a scope (conceptually a file, group, or table)
// and a scope-unique key.
//
// Each provider implementation can handle any number of scope types. Users are
// expected to query for support with iree_io_parameter_provider_query_support
// prior to performing operations.
//
// Thread-safe: a provider may be shared by several contexts simultaneously.
// Behavior is currently undefined if multiple contexts attempt to read or write
// the same parameters concurrently. Future revisions may require that providers
// track pending operations per parameter and sequencing appropriately.
typedef struct iree_io_parameter_provider_t iree_io_parameter_provider_t;

// Retains the given |provider| for the caller.
IREE_API_EXPORT void iree_io_parameter_provider_retain(
    iree_io_parameter_provider_t* provider);

// Releases the given |provider| from the caller.
IREE_API_EXPORT void iree_io_parameter_provider_release(
    iree_io_parameter_provider_t* provider);

// Notifies the provider of an event from the hosting program.
// Providers can ignore notifications at their peril.
IREE_API_EXPORT iree_status_t
iree_io_parameter_provider_notify(iree_io_parameter_provider_t* provider,
                                  iree_io_parameter_provider_signal_t signal);

// Returns true if the given |scope| is supported by |provider|.
IREE_API_EXPORT bool iree_io_parameter_provider_query_support(
    iree_io_parameter_provider_t* provider, iree_string_view_t scope);

// Loads a parameter from |provider| for use on |device|.
// |source_scope| and |source_key| define the parameter and |target_params|
// defines how the buffer is to be allocated.
//
// If the implementation is able to meet the expected |target_params| with an
// existing buffer it may be returned without a new allocation. If access allows
// implementations are allowed to return mapped memory that may be shared by
// other users within the same process or across processes.
//
// Implementations that have no optimized load/import path can implement this
// with iree_hal_device_queue_alloca and iree_io_parameter_provider_read.
//
// Returns IREE_STATUS_NOT_FOUND if the parameter is not found.
IREE_API_EXPORT iree_status_t iree_io_parameter_provider_load(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_params_t target_params,
    iree_device_size_t length,
    iree_hal_buffer_t** IREE_RESTRICT out_target_buffer);

// Reads a parameter from |provider| for use on |device|.
// |source_scope| and |source_key| define the parameter to be read into
// |target_buffer| at |target_offset|.
//
// Returns IREE_STATUS_NOT_FOUND if the parameter is not found.
IREE_API_EXPORT iree_status_t iree_io_parameter_provider_read(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

// Writes a parameter to |provider| from |device|.
// The parameter data is sourced from |source_buffer| at |source_offset| and
// |target_scope| and |target_key| define which parameter is being written.
//
// Returns IREE_STATUS_NOT_FOUND if the parameter is not found.
IREE_API_EXPORT iree_status_t iree_io_parameter_provider_write(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_string_view_t target_scope, iree_string_view_t target_key,
    uint64_t target_offset, iree_device_size_t length);

typedef iree_status_t(IREE_API_PTR* iree_io_parameter_enumerator_fn_t)(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span);

typedef struct iree_io_parameter_enumerator_t {
  // Callback function pointer.
  iree_io_parameter_enumerator_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_io_parameter_enumerator_t;

// Gathers zero or more spans from |provider| into the given |target_buffer|.
// The |enumerator| defines the source keys in |source_scope| and the offset and
// length in the |target_buffer| of each span. Multiple spans may reference the
// same source parameter but behavior is undefined if multiple span target
// ranges overlap.
//
// Returns IREE_STATUS_NOT_FOUND if any parameter is not found.
IREE_API_EXPORT iree_status_t iree_io_parameter_provider_gather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator);

// Scatters zero or more spans to |provider| from the given |source_buffer|.
// The |enumerator| defines the target keys in |target_scope| and the offset and
// length in the |source_buffer| of each span to scatter. Multiple spans may
// reference source ranges that overlap but behavior is undefined if multiple
// spans share the same target parameter.
//
// Returns IREE_STATUS_NOT_FOUND if any parameter is not found.
IREE_API_EXPORT iree_status_t iree_io_parameter_provider_scatter(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator);

//===----------------------------------------------------------------------===//
// iree_io_parameter_provider_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_io_parameter_provider_vtable_t {
  void(IREE_API_PTR* destroy)(
      iree_io_parameter_provider_t* IREE_RESTRICT provider);

  iree_status_t(IREE_API_PTR* notify)(
      iree_io_parameter_provider_t* provider,
      iree_io_parameter_provider_signal_t signal);

  bool(IREE_API_PTR* query_support)(iree_io_parameter_provider_t* provider,
                                    iree_string_view_t scope);

  iree_status_t(IREE_API_PTR* load)(
      iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_string_view_t source_scope, iree_string_view_t source_key,
      uint64_t source_offset, iree_hal_buffer_params_t target_params,
      iree_device_size_t length,
      iree_hal_buffer_t** IREE_RESTRICT out_target_buffer);

  iree_status_t(IREE_API_PTR* read)(
      iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_string_view_t source_scope, iree_string_view_t source_key,
      uint64_t source_offset, iree_hal_buffer_t* target_buffer,
      iree_device_size_t target_offset, iree_device_size_t length);

  iree_status_t(IREE_API_PTR* write)(
      iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_string_view_t target_scope, iree_string_view_t target_key,
      uint64_t target_offset, iree_device_size_t length);

  iree_status_t(IREE_API_PTR* gather)(
      iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
      iree_host_size_t count, iree_io_parameter_enumerator_t enumerator);

  iree_status_t(IREE_API_PTR* scatter)(
      iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
      iree_host_size_t count, iree_io_parameter_enumerator_t enumerator);
} iree_io_parameter_provider_vtable_t;

struct iree_io_parameter_provider_t {
  iree_atomic_ref_count_t ref_count;
  const iree_io_parameter_provider_vtable_t* vtable;
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_PARAMETER_PROVIDER_H_
