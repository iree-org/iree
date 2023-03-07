// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_TRACE_REPLAY_H_
#define IREE_TOOLING_TRACE_REPLAY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_trace_replay_shutdown_flag_bits_e {
  IREE_TRACE_REPLAY_SHUTDOWN_QUIET = 0u,
  IREE_TRACE_REPLAY_SHUTDOWN_PRINT_STATISTICS = 1 << 0u,
};
typedef uint32_t iree_trace_replay_shutdown_flags_t;

typedef struct iree_trace_replay_t {
  iree_allocator_t host_allocator;
  iree_string_view_t root_path;

  iree_vm_instance_t* instance;
  iree_vm_context_flags_t context_flags;

  iree_hal_driver_registry_t* driver_registry;
  iree_host_size_t device_uri_count;
  const iree_string_view_t* device_uris;

  iree_vm_context_t* context;
  iree_hal_device_t* device;
} iree_trace_replay_t;

// Initializes a trace replay context.
// Relative paths will be joined with |root_path| to form a fully-qualified
// path (may be cwd, may be file source, etc).
iree_status_t iree_trace_replay_initialize(
    iree_string_view_t root_path, iree_vm_instance_t* instance,
    iree_vm_context_flags_t context_flags,
    iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, iree_trace_replay_t* out_replay);

// Deinitializes a trace replay context and releases all resources.
void iree_trace_replay_deinitialize(iree_trace_replay_t* replay,
                                    iree_trace_replay_shutdown_flags_t flags);

// TODO(#5724): remove this and instead provide a device set on initialize.
// Overrides the HAL driver used in the trace with the given |driver|.
// |device_uris| must remain valid for the lifetime of the replay instance.
void iree_trace_replay_set_hal_devices_override(
    iree_trace_replay_t* replay, iree_host_size_t device_uri_count,
    const iree_string_view_t* device_uris);

// Replays the given |event_node| against the replay context.
// Automatically switches between the default iree_trace_replay_event_* methods.
iree_status_t iree_trace_replay_event(iree_trace_replay_t* replay,
                                      yaml_document_t* document,
                                      yaml_node_t* event_node);

// Replays a `context_load` event against the replay context.
iree_status_t iree_trace_replay_event_context_load(iree_trace_replay_t* replay,
                                                   yaml_document_t* document,
                                                   yaml_node_t* event_node);

// Replays a `module_load` event against the replay context.
iree_status_t iree_trace_replay_event_module_load(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node);

// Prepares to replay a `call` event.
// |out_function| will contain the function to invoke and |out_input_list| will
// contain the caller-owned inputs to the invocation.
iree_status_t iree_trace_replay_event_call_prepare(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t* out_function,
    iree_vm_list_t** out_input_list);

// Replays a `call` event against the replay context.
// Optionally |out_output_list| can be populated with a caller-owned set of
// outputs from the call.
iree_status_t iree_trace_replay_event_call(iree_trace_replay_t* replay,
                                           yaml_document_t* document,
                                           yaml_node_t* event_node,
                                           iree_vm_list_t** out_output_list);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_TRACE_REPLAY_H_
